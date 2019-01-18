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

        //cuMat
        std::cout << " Run CuMat" << std::endl;
        Json::Array resultsCuMat;
        benchmark_cuMat(parameterNames, params, returnNames, resultsCuMat);
        
        //cuBlas
        std::cout << " Run cuBLAS" << std::endl;
        Json::Array resultsCuBlas;
        benchmark_cuBlas(parameterNames, params, returnNames, resultsCuBlas);

        //Eigen
        std::cout << " Run Eigen" << std::endl;
        Json::Array resultsEigen;
        benchmark_Eigen(parameterNames, params, returnNames, resultsEigen);

        //numpy
        std::cout << " Run Numpy" << std::endl;
        std::string numpyFile = std::string(CUMAT_STR(PYTHON_FILES)) + "Implementation_numpy.py";
        std::string launchParams = "\"" + pythonPath + " " + numpyFile + " " + std::string(CUMAT_STR(CONFIG_FILE)) + " \"" + setName + "\"" + "\"";
        std::cout << "  Args: " << launchParams << std::endl;
        std::string resultsNumpyStr = exec(launchParams.c_str());
        Json::Array resultsNumpy = Json::ParseString(resultsNumpyStr);

        ////tensorflow
        //std::cout << " Run Tensorflow" << std::endl;
        //std::string tfFile = std::string(CUMAT_STR(PYTHON_FILES)) + "Implementation_tensorflow.py";
        //launchParams = pythonPath + " " + tfFile + " " + std::string(CUMAT_STR(CONFIG_FILE)) + " \"" + setName + "\"";
        //std::cout << "  Args: " << launchParams << std::endl;
        //std::string resultsTFStr = exec(launchParams.c_str());
        //Json::Array resultsTF = Json::ParseString(resultsTFStr);

        //write results
        Json::Object resultAssembled;
        resultAssembled.Insert(std::make_pair("CuMat", resultsCuMat));
        resultAssembled.Insert(std::make_pair("CuBlas", resultsCuBlas));
        resultAssembled.Insert(std::make_pair("Eigen", resultsEigen));
        resultAssembled.Insert(std::make_pair("Numpy", resultsNumpy));
        //resultAssembled.Insert(std::make_pair("Tensorflow", resultsTF));
        std::ofstream outStream(outputDir + setName + ".json");
        outStream << resultAssembled;
        outStream.close();
        launchParams = "\"" + pythonPath + " " + std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlots.py" + " \"" + outputDir + setName + "\" " + std::string(CUMAT_STR(CONFIG_FILE)) + "\"";
        std::cout << launchParams << std::endl;
        system(launchParams.c_str());
    }
    std::cout << "DONE" << std::endl;
}
