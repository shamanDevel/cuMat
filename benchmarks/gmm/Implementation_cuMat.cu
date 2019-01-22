#include "benchmark.h"

#include <cuMat/Core>
#include <iostream>
#include <cstdlib>

void benchmark_cuMat(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues)
{
    //number of runs for time measures
    const int runs = 10;
	const int subruns = 10;

    //test if the config is valid
    assert(parameterNames.size() == 2);
    assert(parameterNames[0] == "Vector-Size");
    assert(parameterNames[1] == "Num-Combinations");
    assert(returnNames.size() == 1);
    assert(returnNames[0] == "Time");

    cuMat::SimpleRandom rand;

    int numConfigs = parameters.Size();
    for (int config = 0; config < numConfigs; ++config)
    {
        //Input
        int vectorSize = parameters[config][0].AsInt32();
        int numCombinations = parameters[config][1].AsInt32();
        double totalTime = 0;
        std::cout << "  VectorSize: " << vectorSize << ", Num-Combinations: " << numCombinations << std::flush;

        //Create matrices
        std::vector<cuMat::VectorXf> vectors(numCombinations);
        std::vector<float> factors(numCombinations);
        for (int i = 0; i < numCombinations; ++i) {
            vectors[i] = cuMat::VectorXf(vectorSize);
            rand.fillUniform(vectors[i], 0, 1);
            factors[i] = std::rand() / (float)(RAND_MAX);
        }
		cuMat::VectorXf output(vectorSize);

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {
			output.setZero();

            //Main logic
            cudaDeviceSynchronize();
			auto start = std::chrono::steady_clock::now();

			for (int subrun = 0; subrun < subruns; ++subrun) {
				switch (numCombinations)
				{
				case 1: output.inplace() = (vectors[0] * factors[0]); break;
				case 2: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1]); break;
				case 3: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2]); break;
				case 4: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3]); break;
				case 5: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4]); break;
				case 6: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5]); break;
				case 7: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6]); break;
				case 8: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7]); break;
				case 9: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7] + vectors[8] * factors[8]); break;
				case 10: output.inplace() = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7] + vectors[8] * factors[8] + vectors[9] * factors[9]); break;
				}
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