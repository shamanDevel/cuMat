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
#include <random>

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

	std::string numpyFile, launchParams;

	auto seed = 128;//std::random_device()();

    //start test sets
    const Json::Array& sets = config["Tests"].AsArray();
    for (auto it = sets.Begin(); it != sets.End(); ++it)
    {
        const Json::Array& params = it->AsArray();
		int dimension = params[0].AsInt32();
		int numTrueComponents = params[1].AsInt32();
		int numModelComponents = params[2].AsInt32();
		int numPoints = params[3].AsInt32();
		int numIterations = params[4].AsInt32();
		std::cout << std::endl << "Configuration: dimension=" << dimension
			<< ", numTrueComponents=" << numTrueComponents
			<< ", numModelComponents=" << numModelComponents
			<< ", numPoints=" << numPoints
			<< ", numIterations=" << numIterations << std::endl;

		//create ground truth and test set
		std::cout << " Generate Testset" << std::endl;
		numpyFile = std::string(CUMAT_STR(PYTHON_FILES)) + "GenerateData.py";
		launchParams = "\"" + pythonPath + " " + numpyFile
    		+ " " + std::to_string(dimension) + " " + std::to_string(numTrueComponents) + " " + std::to_string(numPoints) + " " + std::to_string(++seed)
    		+ " GroundTruth.txt \"";
		std::cout << "  Args: " << launchParams << std::endl;
		exec(launchParams.c_str());
		numpyFile = std::string(CUMAT_STR(PYTHON_FILES)) + "GenerateData.py";
		launchParams = "\"" + pythonPath + " " + numpyFile
			+ " " + std::to_string(dimension) + " " + std::to_string(numTrueComponents) + " 0" + " " + std::to_string(++seed)
			+ " Initial.txt \"";
		std::cout << "  Args: " << launchParams << std::endl;
		exec(launchParams.c_str());

		//prepare results
		Json::Object resultAssembled;

        //Eigen
        std::cout << " Run Eigen" << std::endl;
        Json::Object resultsEigen;
        benchmark_Eigen("GroundTruth.txt", "Initial.txt", numIterations, resultsEigen);
		resultAssembled.Insert(std::make_pair("Eigen", resultsEigen));

		//cuMat
		std::cout << " Run cuMat" << std::endl;
		Json::Object resultsCuMat;
		benchmark_cuMat("GroundTruth.txt", "Initial.txt", numIterations, resultsCuMat);
		resultAssembled.Insert(std::make_pair("CuMat", resultsCuMat));

        //write results
        std::ofstream outStream(outputDir + "GMM_Test.json");
        outStream << resultAssembled;
        outStream.close();
        launchParams = "\"" + pythonPath + " " + std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlots1.py" + " GroundTruth.txt Initial.txt " + " \"" + outputDir + "GMM_Test.json" + "\" " + std::string(CUMAT_STR(CONFIG_FILE)) + "\"";
        std::cout << launchParams << std::endl;
        system(launchParams.c_str());
    }

	//start benchmark sets
	const Json::Object benchmarks = config["Benchmarks"].AsObject();
	for (auto it = benchmarks.Begin(); it != benchmarks.End(); ++it)
	{
		const std::string name = it->first;
		const Json::Array& settings = it->second;
		std::cout << "=== Benchmark " << name << " ===" << std::endl;

		Json::Array resultAssembled;
		int runs = settings.Size();
		for (int run = 0; run<runs; ++run)
		{
			Json::Array params = settings[run].AsArray();
			Json::Object currentResult;
			currentResult.Insert(std::make_pair("Settings", params));

			int dimension = params[0].AsInt32();
			int numTrueComponents = params[1].AsInt32();
			int numModelComponents = params[2].AsInt32();
			int numPoints = params[3].AsInt32();
			int numIterations = params[4].AsInt32();

			//create ground truth and test set
			std::cout << " Generate Testset" << std::endl;
			numpyFile = std::string(CUMAT_STR(PYTHON_FILES)) + "GenerateData.py";
			launchParams = "\"" + pythonPath + " " + numpyFile
				+ " " + std::to_string(dimension) + " " + std::to_string(numTrueComponents) + " " + std::to_string(numPoints) + " " + std::to_string(++seed)
				+ " GroundTruth.txt \"";
			std::cout << "  Args: " << launchParams << std::endl;
			exec(launchParams.c_str());
			numpyFile = std::string(CUMAT_STR(PYTHON_FILES)) + "GenerateData.py";
			launchParams = "\"" + pythonPath + " " + numpyFile
				+ " " + std::to_string(dimension) + " " + std::to_string(numTrueComponents) + " 0" + " " + std::to_string(++seed)
				+ " Initial.txt \"";
			std::cout << "  Args: " << launchParams << std::endl;
			exec(launchParams.c_str());

			//Eigen
			std::cout << " Run Eigen" << std::endl;
			Json::Object resultsEigen;
			benchmark_Eigen("GroundTruth.txt", "Initial.txt", numIterations, resultsEigen);
			currentResult.Insert(std::make_pair("Eigen", resultsEigen["Time"]));

			//cuMat
			std::cout << " Run cuMat" << std::endl;
			Json::Object resultsCuMat;
			benchmark_cuMat("GroundTruth.txt", "Initial.txt", numIterations, resultsCuMat);
			currentResult.Insert(std::make_pair("CuMat", resultsCuMat["Time"]));

			resultAssembled.PushBack(currentResult);
		}

		//write results
		std::string outName = outputDir + "GMM_" + name + ".json";
		std::ofstream outStream(outName);
		outStream << resultAssembled;
		outStream.close();
		launchParams = "\"" + pythonPath + " " + std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlots2.py" + " \"" + outName + "\" " + "\"";
		std::cout << launchParams << std::endl;
		system(launchParams.c_str());
	}

    std::cout << "DONE" << std::endl;
}
