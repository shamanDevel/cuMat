#include <stdlib.h>
#include <stdio.h>
#include <cuMat/Core>
#include <random>

using namespace cuMat;

/*
 * This Demo showcases how unary functors can be abused to procedurally generate any data,
 * in this example, the Buddhabrot fractal.
 * This is an extension to BuddhabrotSimple showcasing how a long-running kernel can be split
 * into smaller chunks to prevent the computer from freezing and to prevent timeout errors.
 */

//CONFIGURATION
constexpr int WIDTH = 1920;
constexpr int HEIGHT = 1080;
constexpr double RANGE = 2.0; // from -2 to 2 in the reals
constexpr double OFFSET_REAL = -0.5;
constexpr double ESCAPE_RADIUS_SQ = 4.0 * 4.0;
constexpr int ITERATIONS[3] = { 10000, 1000, 100 }; //red, green, blue
constexpr int ITERATIONS_PER_STEP = 32;
constexpr int MSAA_SQR = 4;
constexpr int MSAA = MSAA_SQR * MSAA_SQR;

//output types
typedef Matrix<int, MSAA, Dynamic, Dynamic, RowMajor> IterationMatrix;
typedef Matrix<cdouble, MSAA, Dynamic, Dynamic, RowMajor> PositionMatrix;
typedef Matrix<int, Dynamic, Dynamic, 1, ColumnMajor> OutputMatrix;

//output / helper
void writePPM(const OutputMatrix& red, const OutputMatrix& green, const OutputMatrix& blue, const char const* filename);
void printProgress(double percentage);

// Sampling pattern (in pixels) for jittered sampling
__constant__ double2 samplingPattern[MSAA];
// Generates the sampling pattern
void generateSamplingPattern()
{
	double2 pattern[MSAA];
	std::default_random_engine rnd;
	std::uniform_real_distribution<double> distr(-0.5 / MSAA_SQR, +0.5 / MSAA_SQR);
	const double step = 1.0 / MSAA_SQR;
	const double offset = -0.5 + step / 2;
	for (int i=0; i<MSAA_SQR; ++i)
		for (int j=0; j<MSAA_SQR; ++j)
		{
			pattern[i + MSAA_SQR * j].x = offset + step * i + distr(rnd);
			pattern[i + MSAA_SQR * j].y = offset + step * j + distr(rnd);
			//printf(" (%6.4f, %6.4f)\n", float(pattern[i + MSAA_SQR * j].x), float(pattern[i + MSAA_SQR * j].y));
		}
	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(samplingPattern, pattern, sizeof(double2)*MSAA));
}

//main functor
//computes the Buddhabrot paths.
//Because the buddhabrot can't be computed in one step for large iteration counts without time-out,
//it is computed iteratively. Hence, this functor is a unary functor that updates the current 
//position z_n in every step while keeping trace of the number of iterations.
template<bool FirstPass>
struct BuddhaBrotFunctor
{
private:
	mutable OutputMatrix buddhabrot_;
	mutable IterationMatrix iterationMatrix_;
	const double width_;
	const double height_;
	const double range_;
	const double step_;
	const double offsetReal_;
	const int iterations_;
	const double escapeRadiusSq_;
public:
	BuddhaBrotFunctor(OutputMatrix buddhabrot, IterationMatrix iterationMatrix, 
		int width, int height, double range, double offsetReal, int iterations, double escapeRadiusSq)
		: buddhabrot_(buddhabrot),
		iterationMatrix_(iterationMatrix),
		width_(width), height_(height),
		range_(range), step_(width / (2.0 * range)),
		offsetReal_(offsetReal),
		iterations_(iterations), escapeRadiusSq_(escapeRadiusSq)
	{}

	typedef cdouble ReturnType;
	__device__ CUMAT_STRONG_INLINE ReturnType operator()(cdouble z, Index row, Index col, Index batch) const
	{
		const int msaa = row;
		const int x = col;
		const int y = batch;
		//read iteration matrix
		const int startIteration = iterationMatrix_.coeff(msaa, x, y, -1);
		//the start position in the complex plane
		const cdouble c(
			(x + samplingPattern[msaa].x - width_*0.5) / step_ + offsetReal_,
			(y + samplingPattern[msaa].y - height_*0.5) / step_);

		if (FirstPass) {
			//first run, check if the mandelbulb series diverges
			if (startIteration < 0)
				return z; //already diverged
			int i;
			for (i = 0; i < iterations_; ++i)
			{
				z = z * z + c;
				if (z.real()*z.real() + z.imag()*z.imag() > escapeRadiusSq_)
					break;
			}
			if (i < iterations_)
				iterationMatrix_.coeff(msaa, x, y, -1) = -startIteration - i; //diverged
			else
				iterationMatrix_.coeff(msaa, x, y, -1) = startIteration + iterations_;
		}
		else
		{
			//second run, trace all points again that are NOT in the Mandelbulb.
			//This means, startIteration is negative and the counter indicates how many points to trace
			if (startIteration >= 0)
				return z; //inside the Mandelbulb or done tracing
			const int toTrace = min(iterations_, -startIteration);
			//trace points and add to map
			for (int i=0; i<toTrace; ++i)
			{
				z = z * z + c;
				int px = (z.real() - offsetReal_)*step_ + width_ * 0.5;
				int py = z.imag()*step_ + height_ * 0.5;
				if (px >= 0 && py >= 0 && px < width_ && py < height_)
					atomicAdd(buddhabrot_.data() + buddhabrot_.index(px, py, 0), 1);
			}
			//update number of iterations to trace
			iterationMatrix_.coeff(msaa, x, y, -1) = startIteration + toTrace;
		}
		//return current trace position
		return z;
	}
};

int main()
{
	generateSamplingPattern();
	//compute mandelbrot and buddhabrot, three times
	OutputMatrix buddhabrot[3];
	int kernelCallCounter = 0;
	for (int i=0; i<3; ++i)
	{
		int iterations = ITERATIONS[i];
		printf("Compute Buddhabrot with %d iterations\n", iterations);
		buddhabrot[i] = MatrixXi::Zero(WIDTH, HEIGHT);
		IterationMatrix iterationMatrix = IterationMatrix::Zero(MSAA, WIDTH, HEIGHT);
		PositionMatrix positionMatrix = PositionMatrix::Zero(MSAA, WIDTH, HEIGHT);
		//first pass
		BuddhaBrotFunctor<true> functor1(buddhabrot[i], iterationMatrix, WIDTH, HEIGHT, RANGE, OFFSET_REAL, ITERATIONS_PER_STEP, ESCAPE_RADIUS_SQ);
		for (int j=0; j<iterations; j+=ITERATIONS_PER_STEP)
		{
			//print and sync at every few kernel calls
			//otherwise, your computer will freeze because of 100% GPU load.
			if ((kernelCallCounter++) % 2 == 0) {
				CUMAT_SAFE_CALL(cudaDeviceSynchronize());
				printProgress(j / double(iterations));
			}
			positionMatrix.inplace() = positionMatrix.unaryExpr(functor1);
		}
		printProgress(1.0); printf("\n");
		//second pass
		positionMatrix.setZero();
		BuddhaBrotFunctor<false> functor2(buddhabrot[i], iterationMatrix, WIDTH, HEIGHT, RANGE, OFFSET_REAL, ITERATIONS_PER_STEP, ESCAPE_RADIUS_SQ);
		for (int j = 0; j < iterations; j += ITERATIONS_PER_STEP)
		{
			//print and sync at every few kernel calls
			//otherwise, your computer will freeze because of 100% GPU load.
			if ((kernelCallCounter++) % 2 == 0) {
				CUMAT_SAFE_CALL(cudaDeviceSynchronize());
				printProgress(j / double(iterations));
			}
			positionMatrix.inplace() = positionMatrix.unaryExpr(functor2);
		}
		printProgress(1.0); printf("\n");
	}

	//save them
	printf("Save to file\n");
	writePPM(buddhabrot[0], buddhabrot[1], buddhabrot[2], "BuddhabrotLarge.ppm");
}

//Source: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file#C
void writePPM(const OutputMatrix& red, const OutputMatrix& green, const OutputMatrix& blue, const char const* filename)
{
	//normalize and convert to uint8
	Matrix<uint8_t, Dynamic, Dynamic, 1, ColumnMajor> byteRed = ((red.cast<float>()
		/ static_cast<float>(static_cast<int>(red.maxCoeff())))
		*255 ).cast<uint8_t>();
	Matrix<uint8_t, Dynamic, Dynamic, 1, ColumnMajor> byteGreen = ((green.cast<float>()
		/ static_cast<float>(static_cast<int>(green.maxCoeff())))
		* 255).cast<uint8_t>();
	Matrix<uint8_t, Dynamic, Dynamic, 1, ColumnMajor> byteBlue = ((blue.cast<float>()
		/ static_cast<float>(static_cast<int>(blue.maxCoeff())))
		* 255).cast<uint8_t>();

	//copy to CPU
	int dimx = byteRed.rows();
	int dimy = byteRed.cols();
	std::vector<uint8_t> dataRed(dimx * dimy);
	byteRed.copyToHost(&dataRed[0]);
	std::vector<uint8_t> dataGreen(dimx * dimy);
	byteBlue.copyToHost(&dataGreen[0]);
	std::vector<uint8_t> dataBlue(dimx * dimy);
	byteBlue.copyToHost(&dataBlue[0]);

	//save PGM
	FILE *fp = fopen(filename, "wb"); /* b - binary mode */
	fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
	for (size_t i=0; i<dataRed.size(); ++i)
	{
		static unsigned char color[3];
		color[0] = dataRed[i];
		color[1] = dataGreen[i];
		color[2] = dataBlue[i];
		(void)fwrite(color, 1, 3, fp);
	}
	fclose(fp);
}

//Source: https://stackoverflow.com/a/36315819
#define PBSTR "############################################################"
#define PBWIDTH 60
void printProgress(double percentage)
{
	int val = (int)(percentage * 100);
	int lpad = (int)(percentage * PBWIDTH);
	int rpad = PBWIDTH - lpad;
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}