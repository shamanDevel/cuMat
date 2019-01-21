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
    assert(parameterNames.size() == 1);
    assert(parameterNames[0] == "Vector-Size");
    assert(returnNames.size() == 1);
    assert(returnNames[0] == "Time");

    cuMat::SimpleRandom rand;

    int numConfigs = parameters.Size();
    for (int config = 0; config < numConfigs; ++config)
    {
        //Input
        int vectorSize = parameters[config][0].AsInt32();
        double totalTime = 0;
        std::cout << "  VectorSize: " << vectorSize << std::flush;

        //Create matrices
		cuMat::VectorXf a(vectorSize); rand.fillUniform(a, 0, 1);
		cuMat::VectorXf b(vectorSize); rand.fillUniform(b, 0, 1);

		volatile float res;
        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {

            //Main logic
            cudaDeviceSynchronize();
			auto start = std::chrono::steady_clock::now();

			for (int i=0; i<subruns; ++i)
			{
				res = static_cast<float>(a.dot(b));
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