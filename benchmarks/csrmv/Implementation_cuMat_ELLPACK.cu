#include "benchmark.h"

#include <Eigen/Sparse>
#include <cuMat/Core>
#include <cuMat/Sparse>
#include <iostream>
#include <cstdlib>

void benchmark_cuMat_ELLPACK(
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
		int nnzPerRow = 5;
		Eigen::Matrix<int, Eigen::Dynamic, 5, Eigen::ColMajor> indices(matrixSize, 5); indices.fill(-1);
		Eigen::Matrix<float, Eigen::Dynamic, 5, Eigen::ColMajor> values(matrixSize, 5);
		for (int x = 0; x<gridSize; ++x) for (int y = 0; y<gridSize; ++y)
		{
			int row = IDX(x, y);
			int ci = 0;
			if (x > 0) {
				indices(row, ci) = IDX(x - 1, y);
				values(row, ci) = -1;
				ci++;
			}
			if (y > 0) {
				indices(row, ci) = IDX(x, y - 1);
				values(row, ci) = -1;
				ci++;
			}
			{
				indices(row, ci) = row;
				values(row, ci) = 4;
				ci++;
			}
			if (y < gridSize - 1) {
				indices(row, ci) = IDX(x, y + 1);
				values(row, ci) = -1;
				ci++;
			}
			if (x < gridSize - 1) {
				indices(row, ci) = IDX(x + 1, y);
				values(row, ci) = -1;
				ci++;
			}
		}

		//Create vector
		Eigen::VectorXf ex = Eigen::VectorXf::Random(matrixSize);

		//Send to cuMat
		typedef cuMat::SparseMatrix<float, 1, cuMat::ELLPACK> SMatrix;
		typedef cuMat::SparsityPattern<cuMat::ELLPACK> SPattern;
		SPattern pattern;
		pattern.rows = matrixSize;
		pattern.cols = matrixSize;
		pattern.nnzPerRow = nnzPerRow;
		pattern.indices = SPattern::IndexMatrix::fromEigen(indices);
        pattern.assertValid();
		SMatrix mat(pattern);
		mat.getData() = SMatrix::DataMatrix::fromEigen(values);

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