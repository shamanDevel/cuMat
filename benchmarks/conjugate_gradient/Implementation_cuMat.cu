#include "benchmark.h"

#include <Eigen/Sparse>
#include <cuMat/Core>
#include <cuMat/Sparse>
#include <cuMat/src/ConjugateGradient.h>
#include <iostream>
#include <cstdlib>

void benchmark_cuMat(
	const std::vector<std::string>& parameterNames,
	const Json::Array& parameters,
	const std::vector<std::string>& returnNames,
	Json::Array& returnValues)
{
	//number of runs for time measures
	const int runs = 2;

	int numConfigs = parameters.Size();
	for (int config = 0; config < numConfigs; ++config)
	{
		//Input
		int gridSize = parameters[config][0].AsInt32();
		double totalTime = 0;
		std::cout << "  Grid Size: " << gridSize << std::flush;
		int matrixSize = gridSize * gridSize;

		//Create matrix
#define IDX(x, y) ((y) + (x)*gridSize)
		typedef Eigen::SparseMatrix<float, Eigen::RowMajor> ESMatrix;
		ESMatrix matrix(matrixSize, matrixSize);
		matrix.reserve(Eigen::VectorXi::Constant(matrixSize, 5));
		for (int x = 0; x < gridSize; ++x) for (int y = 0; y < gridSize; ++y)
		{
			int row = IDX(x, y);
			if (x > 0) matrix.insert(row, IDX(x - 1, y)) = -1;
			if (y > 0) matrix.insert(row, IDX(x, y - 1)) = -1;
			matrix.insert(row, row) = 4;
			if (y < gridSize - 1) matrix.insert(row, IDX(x, y + 1)) = -1;
			if (x < gridSize - 1) matrix.insert(row, IDX(x + 1, y)) = -1;
		}
		matrix.makeCompressed();

		//Create vector
		Eigen::VectorXf erhs = Eigen::VectorXf::Zero(matrixSize);
		std::srand(42);
		for (int x = 0; x < gridSize; ++x) for (int y = 0; y < gridSize; ++y) {
			if (x == 0 || y == 0 || x == gridSize - 1 || y == gridSize - 1)
				erhs[IDX(x, y)] = std::rand() / float(RAND_MAX);
		}

		//Send to cuMat
		typedef cuMat::SparseMatrix<float, 1, cuMat::CSR> SMatrix;
		typedef cuMat::SparsityPattern<cuMat::CSR> SPattern;
		SPattern pattern;
		pattern.rows = matrixSize;
		pattern.cols = matrixSize;
		pattern.nnz = matrix.nonZeros();
		pattern.JA = SPattern::IndexVector(matrixSize + 1); pattern.JA.copyFromHost(matrix.outerIndexPtr());
		pattern.IA = SPattern::IndexVector(pattern.nnz); pattern.IA.copyFromHost(matrix.innerIndexPtr());
        pattern.assertValid();
		SMatrix mat(pattern);
		mat.getData().copyFromHost(matrix.valuePtr());

		cuMat::VectorXf rhs = cuMat::VectorXf::fromEigen(erhs);
		cuMat::VectorXf r = cuMat::VectorXf::Zero(matrixSize);

		//Run it multiple times
		int iterations = 0; float error = 0;
		for (int run = 0; run < runs; ++run)
		{
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			//Main logic
			cudaDeviceSynchronize();
			//cudaEventRecord(start, cuMat::Context::current().stream());
			auto start2 = std::chrono::steady_clock::now();
			cuMat::ConjugateGradient<SMatrix> cg(mat);
			cg.setTolerance(1e-4);
			r.inplace() = cg.solve(rhs);

			//cudaEventRecord(stop, cuMat::Context::current().stream());
			//cudaEventSynchronize(stop);
			//float elapsed;
			//cudaEventElapsedTime(&elapsed, start, stop);

			cudaDeviceSynchronize();
			auto finish2 = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration_cast<
				std::chrono::duration<double> >(finish2 - start2).count() * 1000;

			iterations = cg.iterations();
			error = cg.error();

			totalTime += elapsed;
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}

		//Result
		Json::Array resultJson;
		double finalTime = totalTime / runs;
		resultJson.PushBack(finalTime);
		returnValues.PushBack(resultJson);
		std::cout << " -> " << finalTime << "ms" << " (iterations=" << iterations << ", error=" << error << ")" << std::endl;
	}
}