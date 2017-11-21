/*
 * Launches the benchmarks.
 * The path to the config file is defined in the macro CONFIG_FILE
 */

#include <iostream>
#include <vector>
#include <string>

#include "json_st.h"
#include "Json.h"
#include "benchmark.h"
#include <cuMat/src/Macros.h>

int main(int argc, char* argv[])
{
    //load json
    Json::Object config = Json::ParseFile(std::string(CUMAT_STR(CONFIG_FILE)));
    std::cout << "Start Benchmark '" << config["Title"].AsString() << "'" << std::endl;

    //parse parameter + return names
    std::vector<std::string> parameterNames;
    auto parameterArray = config["Parameters"].AsArray();
    for (auto it = parameterArray.Begin(); it != parameterArray.End(); ++it)
    {
        parameterNames.push_back(it->AsString());
    }
    std::vector<std::string> returnNames;
    auto returnArray = config["Returns"].AsArray();
    for (auto it = returnArray.Begin(); it != returnArray.End(); ++it)
    {
        returnNames.push_back(it->AsString());
    }

    //start test sets
    const Json::Object& sets = config["Sets"].AsObject();
    for (auto it = sets.Begin(); it != sets.End(); ++it)
    {
        std::string setName = it->first;
        const Json::Array& params = it->second.AsArray();
        std::cout << std::endl << "Test Set '" << setName << "'" << std::endl;

        //cuMat
        std::cout << " Run CuMat" << std::endl;
        Json::Array resultsCuMat;
        benchmark_cuMat(parameterNames, params, returnNames, resultsCuMat);

        //Eigen
        std::cout << " Run Eigen" << std::endl;
        Json::Array resultsEigen;
        benchmark_Eigen(parameterNames, params, returnNames, resultsEigen);

        //TODO: write results

    }
}
