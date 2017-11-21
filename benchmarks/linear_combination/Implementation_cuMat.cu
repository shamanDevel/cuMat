#include "../benchmark.h"

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
    const int runs = 1;

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

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            //Main logic
            cudaDeviceSynchronize();
            cudaEventRecord(start, cuMat::Context::current().stream());

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

            cudaEventRecord(stop, cuMat::Context::current().stream());
            cudaEventSynchronize(stop);
            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            totalTime += elapsed;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        //Result
        Json::Array result;
        double finalTime = totalTime / runs;
        result.PushBack(finalTime);
        returnValues.PushBack(result);
        std::cout << " -> " << finalTime << "ms" << std::endl;
    }
}