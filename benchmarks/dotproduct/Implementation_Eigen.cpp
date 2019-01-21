#include "benchmark.h"

#include <Eigen/Core>
#include <chrono>
#include <iostream>
#include <cstdlib>

double dontOptimizeAway = 0;

void benchmark_Eigen(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues)
{
    //number of runs for time measures
    const int runs = 10;
	const int subruns = 1;

    //test if the config is valid
    assert(parameterNames.size() == 1);
    assert(parameterNames[0] == "Vector-Size");
    assert(returnNames.size() == 1);
    assert(returnNames[0] == "Time");

    int numConfigs = parameters.Size();
    for (int config = 0; config < numConfigs; ++config)
    {
        //Input
        int vectorSize = parameters[config][0].AsInt32();
        double totalTime = 0;
        std::cout << "  VectorSize: " << vectorSize << std::flush;

        //Create matrices
		Eigen::VectorXf a(vectorSize); a.setRandom();
		Eigen::VectorXf b(vectorSize); b.setRandom();

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {
            //Main logic
            auto start = std::chrono::steady_clock::now();

			for (int i = 0; i < subruns; ++i) {
				float result = a.dot(b);
				dontOptimizeAway += result;
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