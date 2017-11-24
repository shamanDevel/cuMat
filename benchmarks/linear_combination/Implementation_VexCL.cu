#include "../benchmark.h"

#define VEXCL_BACKEND CUDA
#define VEXCL_BACKEND_CUDA
#include <boost/proto/proto.hpp>
#include <vexcl/backend.hpp>
#include <vexcl/devlist.hpp>
#include <vexcl/constants.hpp>
#include <vexcl/element_index.hpp>
#include <vexcl/vector.hpp>
#include <vexcl/vector_view.hpp>
#include <vexcl/vector_pointer.hpp>
#include <vexcl/tagged_terminal.hpp>
#include <vexcl/temporary.hpp>
#include <vexcl/random.hpp>
#include <vexcl/generator.hpp>
#include <vexcl/profiler.hpp>
#include <vexcl/function.hpp>
#include <vexcl/logical.hpp>
#include <vexcl/eval.hpp>

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

void benchmark_VexCL(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues)
{
    vex::Context ctx(vex::Filter::GPU);
    if (!ctx) throw std::runtime_error("No devices available.");

    //number of runs for time measures
    const int runs = 10;

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
        std::vector<vex::vector<float>> vectors(numCombinations);
        std::vector<float> factors(numCombinations);
        for (int i = 0; i < numCombinations; ++i) {
            vectors[i] = vex::vector<float>(vectorSize);
            vex::Random<double, vex::random::threefry> rnd;
            vectors[i] = rnd(vex::element_index(), std::rand());
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
            cudaEventRecord(start);

            if (numCombinations == 1) {
                vex::vector<float> v = (vectors[0] * factors[0]);
            } 
            else if (numCombinations == 2) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1]);
            } 
            else if (numCombinations == 3) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2]);
            }
            else if (numCombinations == 4) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3]);
            }
            else if (numCombinations == 5) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4]);
            }
            else if (numCombinations == 6) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5]);
            }
            else if (numCombinations == 7) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6]);
            }
            else if (numCombinations == 8) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7]);
            }
            else if (numCombinations == 9) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7] + vectors[8] * factors[8]);
            }
            else if (numCombinations == 10) {
                vex::vector<float> v = (vectors[0] * factors[0] + vectors[1] * factors[1] + vectors[2] * factors[2] + vectors[3] * factors[3] + vectors[4] * factors[4] + vectors[5] * factors[5] + vectors[6] * factors[6] + vectors[7] * factors[7] + vectors[8] * factors[8] + vectors[9] * factors[9]);
            }

            cudaEventRecord(stop);
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