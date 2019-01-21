/*
 * Launches the benchmarks.
 * The path to the config file is defined in the macro CONFIG_FILE
 */

#ifdef _MSC_VER
#include <stdio.h>
#endif

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <fstream>

#include "../json_st.h"
#include "../Json.h"
#include "benchmark.h"
#include <cuMat/src/Macros.h>

//https://stackoverflow.com/a/478960/4053176
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
#ifdef _MSC_VER
    std::shared_ptr<FILE> pipe(_popen(cmd, "rt"), _pclose);
#else
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
#endif
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

int main(int argc, char* argv[])
{
	std::string pythonPath = "\"C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python36_64/python.exe\"";
    std::string outputDir = CUMAT_STR(OUTPUT_DIR);

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
		Json::Object resultAssembled;

        //cuMat
        std::cout << " Run CuMat" << std::endl;
        Json::Array resultsCuMat;
        benchmark_cuMat(parameterNames, params, returnNames, resultsCuMat);
		resultAssembled.Insert(std::make_pair("CuMat", resultsCuMat));
        
		//CUB
		std::cout << " Run CUB" << std::endl;
		Json::Array resultsCub;
		benchmark_CUB(parameterNames, params, returnNames, resultsCub);
		resultAssembled.Insert(std::make_pair("CUB", resultsCub));

		//Thrust
		std::cout << " Run Thrust" << std::endl;
		Json::Array resultsThrust;
		benchmark_Thrust(parameterNames, params, returnNames, resultsThrust);
		resultAssembled.Insert(std::make_pair("Thrust", resultsThrust));

        //cuBlas
        std::cout << " Run cuBLAS" << std::endl;
        Json::Array resultsCuBlas;
        benchmark_cuBlas(parameterNames, params, returnNames, resultsCuBlas);
		resultAssembled.Insert(std::make_pair("CuBlas", resultsCuBlas));

        //Eigen
        std::cout << " Run Eigen" << std::endl;
        Json::Array resultsEigen;
        benchmark_Eigen(parameterNames, params, returnNames, resultsEigen);
		resultAssembled.Insert(std::make_pair("Eigen", resultsEigen));

        //write results
        std::ofstream outStream(outputDir + setName + ".json");
        outStream << resultAssembled;
        outStream.close();
		std::string launchParams = "\"" + pythonPath + " " + std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlots.py" + " \"" + outputDir + setName + "\" " + std::string(CUMAT_STR(CONFIG_FILE)) + "\"";
        std::cout << launchParams << std::endl;
        system(launchParams.c_str());
    }
    std::cout << "DONE" << std::endl;
}
