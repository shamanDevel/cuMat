#include <stdlib.h>
#include <stdio.h>
#include <cuMat/Core>

using namespace cuMat;

/*
 * This Demo showcases how nullary functors can be abused to procedurally generate any data,
 * in this example, the Mandelbulb fractal.
 * As a bonus, this demo also shows how to atomically modify other matrices inside those functors
 * to generate the Buddhabrot fractal.
 */

//CONFIGURATION
const int WIDTH = 500;
const int HEIGHT = 300;
const double RANGE = 2.0; // from -2 to 2 in the reals
const double OFFSET_REAL = -0.5;
const double ESCAPE_RADIUS_SQ = 4.0 * 4.0;
const int ITERATIONS = 100; //red, green, blue

//output
void writePGM(const MatrixXi& intensities, const char const* filename);

//main functor
//computes the Buddhabrot paths and as a side-effect, also the mandelbrot
struct BuddhaBrotFunctor
{
private:
	mutable MatrixXi buddhabrot_;
	const double width_;
	const double height_;
	const double range_;
	const double step_;
	const double offsetReal_;
	const int iterations_;
	const double escapeRadiusSq_;
public:
	BuddhaBrotFunctor(MatrixXi buddhabrot, int width, int height, double range, double offsetReal, int iterations, double escapeRadiusSq)
		: buddhabrot_(buddhabrot),
		width_(width), height_(height),
		range_(range), step_(width / (2.0 * range)),
		offsetReal_(offsetReal),
		iterations_(iterations), escapeRadiusSq_(escapeRadiusSq)
	{}

	typedef int ReturnType;
	__device__ CUMAT_STRONG_INLINE ReturnType operator()(Index row, Index col, Index batch) const
	{
		//the start position in the complex plane
		const cdouble c(
			(row - width_*0.5) / step_ + offsetReal_,
			(col - height_*0.5) / step_);
		//first run, check if the mandelbulb series diverges
		int i;
		cdouble z(0, 0);
		for (i=0; i<iterations_; ++i)
		{
			z = z * z + c;
			if (z.real()*z.real()+z.imag()*z.imag() > escapeRadiusSq_)
				break;
		}
		if (i == iterations_)
		{
			//inside the mandelbulb -> ignore in Buddhabrot
			return 0;
		}
		//Outside of the mandelbulb -> consider in Buddhabrot
		//trace points again, but this time, add to map
		z = cdouble(0, 0);
		for (int i2 = 0; i2 < i; ++i2)
		{
			z = z * z + c;
			int x = (z.real()-offsetReal_)*step_ + width_ * 0.5;
			int y = z.imag()*step_ + height_ * 0.5;
			if (x >= 0 && y >= 0 && x < width_ && y < height_)
				atomicAdd(buddhabrot_.data() + buddhabrot_.index(x, y, 0), 1);
		}
		return i;
	}
};

int main()
{
	//compute mandelbrot and buddhabrot, three times
	printf("Compute Mandelbrot and Buddhabrot with %d iterations\n", ITERATIONS);
	MatrixXi buddhabrot = MatrixXi::Zero(WIDTH, HEIGHT);
	BuddhaBrotFunctor functor(buddhabrot, WIDTH, HEIGHT, RANGE, OFFSET_REAL, ITERATIONS, ESCAPE_RADIUS_SQ);
	MatrixXi mandelbrot = MatrixXi::NullaryExpr<BuddhaBrotFunctor>(WIDTH, HEIGHT, 1, functor);

	//save them
	printf("Save to file\n");
	writePGM(mandelbrot, "MandelbrotSmall.pgm");
	writePGM(buddhabrot, "BuddhabrotSmall.pgm");
}

//Source: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file#C
void writePGM(const MatrixXi& intensities, const char const* filename)
{
	//normalize and convert to uint8
	MatrixXf normalizedIntensitites = intensities.cast<float>() 
		/ static_cast<float>(static_cast<int>(intensities.maxCoeff()));
	Matrix<uint8_t, Dynamic, Dynamic, 1, ColumnMajor> byteIntensitites =
		(normalizedIntensitites * 255).cast<uint8_t>();

	//copy to CPU
	int dimx = byteIntensitites.rows();
	int dimy = byteIntensitites.cols();
	std::vector<uint8_t> data(dimx * dimy);
	byteIntensitites.copyToHost(&data[0]);

	//save PGM
	FILE *fp = fopen(filename, "wb"); /* b - binary mode */
	fprintf(fp, "P5\n%d %d\n255\n", dimx, dimy);
	fwrite(data.data(), sizeof(uint8_t), data.size(), fp);
	fclose(fp);
}
