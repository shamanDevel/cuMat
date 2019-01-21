#include "benchmark.h"

#include <Eigen/Sparse>
#include <cuMat/Core>
#include <cuMat/Sparse>
#include <iostream>
#include <cstdlib>

void benchmark_cuMat_CSR(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues)
{
    //number of runs for time measures
    const int runs = 10;
	const int subruns = 10;

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
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> matrix(matrixSize, matrixSize);
		matrix.reserve(Eigen::VectorXi::Constant(matrixSize, 5));
		for (int x = 0; x<gridSize; ++x) for (int y = 0; y<gridSize; ++y)
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
		Eigen::VectorXf ex = Eigen::VectorXf::Random(matrixSize);

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

		cuMat::VectorXf x = cuMat::VectorXf::fromEigen(ex);
		cuMat::VectorXf r(matrixSize);

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {
            //Main logic
			cudaDeviceSynchronize();
			auto start = std::chrono::steady_clock::now();

			for (int i = 0; i < subruns; ++i) {
				r.inplace() = mat * x;
			}

			cudaDeviceSynchronize();
			auto finish = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration_cast<
				std::chrono::duration<double>>(finish - start).count() * 1000 / subruns;

            totalTime += elapsed;
        }

        //Result
        Json::Array result;
        double finalTime = totalTime / runs;
        result.PushBack(finalTime);
        returnValues.PushBack(result);
        std::cout << " -> " << finalTime << "ms" << std::endl;
    }
}