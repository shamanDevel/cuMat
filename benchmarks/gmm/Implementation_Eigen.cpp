#include "benchmark.h"

#include <Eigen/Core>
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
    const int runs = 10;
	const int subruns = 2;

    //test if the config is valid
    assert(parameterNames.size() == 2);
    assert(parameterNames[0] == "Vector-Size");
    assert(parameterNames[1] == "Num-Combinations");
    assert(returnNames.size() == 1);
    assert(returnNames[0] == "Time");

    int numConfigs = parameters.Size();
    for (int config = 0; config < numConfigs; ++config)
    {
        //Input
        int vectorSize = parameters[config][0].AsInt32();
        int numCombinations = parameters[config][1].AsInt32();
        double totalTime = 0;
        std::cout << "  VectorSize: " << vectorSize << ", Num-Combinations: " << numCombinations << std::flush;

        //Create matrices
        std::vector<Eigen::VectorXf> vectors(numCombinations);
        std::vector<float> factors(numCombinations);
        for (int i = 0; i < numCombinations; ++i) {
            vectors[i] = Eigen::VectorXf(vectorSize);
            vectors[i].setRandom();
            factors[i] = std::rand() / (float)(RAND_MAX);
        }

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {
            //Main logic
            auto start = std::chrono::steady_clock::now();

			for (int subrun = 0; subrun < subruns; ++subrun) {
				switch (numCombinations)
				{
				case 1: (vectors[0] * factors[0]).eval(); break;
				case 2: (vectors[0] * factors[0] + vectors[1] * factors[1]).eval(); break;
				case 3: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2]).eval(); break;
				case 4: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3]).eval(); break;
				case 5: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4]).eval(); break;
				case 6: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5]).eval(); break;
				case 7: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6]).eval(); break;
				case 8: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7]).eval(); break;
				case 9: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7] + vectors[8] * factors[8]).eval(); break;
				case 10: (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7] + vectors[8] * factors[8] + vectors[9] * factors[9]).eval(); break;
				}
			}

            auto finish = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<
                std::chrono::duration<double> >(finish - start).count() * 1000 / subruns;
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