#include "benchmark.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <cstdlib>

void benchmark_Eigen(
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
		Eigen::VectorXf rhs = Eigen::VectorXf::Zero(matrixSize);
		std::srand(42);
		for (int x = 0; x < gridSize; ++x) for (int y = 0; y < gridSize; ++y) {
			if (x == 0 || y == 0 || x == gridSize - 1 || y == gridSize - 1)
				rhs[IDX(x, y)] = std::rand() / float(RAND_MAX);
		}
		Eigen::VectorXf result = Eigen::VectorXf::Zero(matrixSize);

        //Run it multiple times
		int iterations = 0; float error = 0;
        for (int run = 0; run < runs; ++run)
        {
            //Main logic
            auto start = std::chrono::steady_clock::now();

			Eigen::ConjugateGradient<ESMatrix> cg(matrix);
			cg.setTolerance(1e-4);
			result.noalias() = cg.solve(rhs);

            auto finish = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<
                std::chrono::duration<double> >(finish - start).count() * 1000;
            totalTime += elapsed;

			iterations = cg.iterations();
			error = cg.error();
        }

        //Result
        Json::Array resultJson;
        double finalTime = totalTime / runs;
		resultJson.PushBack(finalTime);
        returnValues.PushBack(resultJson);
		std::cout << " -> " << finalTime << "ms" << " (iterations=" << iterations << ", error=" << error << ")" << std::endl;
    }
}