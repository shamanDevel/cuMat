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
#include <exception>
#include <fstream>

#include "../json_st.h"
#include "../Json.h"

#include "reductions.h"



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

Json::Array timingsToArray(int N, const Timings& timings)
{
	Json::Array a;
	a.PushBack(N);
	a.PushBack(timings.baseline);
	a.PushBack(timings.thread);
	a.PushBack(timings.warp);
	a.PushBack(timings.block64);
	a.PushBack(timings.block128);
	a.PushBack(timings.block256);
	a.PushBack(timings.block512);
	a.PushBack(timings.device8);
	a.PushBack(timings.device16);
	a.PushBack(timings.device32);
	return a;
}

int main(int argc, char* argv[])
{
	std::string pythonPath = "\"C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python36_64/python.exe\"";
    std::string outputDir = CUMAT_STR(OUTPUT_DIR);

	//load json
	Json::Object config = Json::ParseFile(std::string(CUMAT_STR(CONFIG_FILE)));

	//validation
	std::cout << "Validate algorithms" << std::endl;
	Json::Array validation = config["Validation"].AsArray();
	for (int i=0; i<validation.Size(); ++i)
	{
		Json::Array sizes = validation[i].AsArray();
		int rows = sizes[0].AsInt32();
		int cols = sizes[1].AsInt32();
		int batches = sizes[2].AsInt32();
		benchmark(rows, cols, batches, "int", "row", true);
		benchmark(rows, cols, batches, "int", "col", true);
		benchmark(rows, cols, batches, "int", "batch", true);
	}

	//benchmarks
	int sizePower = 22;
	int minN = 2;
	int maxN = 1 << (sizePower-2);
	int size = 1 << sizePower;
	int runs = 2;
	Json::Object results;
	results.Insert(std::make_pair("Size", size));

	Json::Array resultsInfo;
	resultsInfo.PushBack("N");
	resultsInfo.PushBack("Baseline");
	resultsInfo.PushBack("Thread");
	resultsInfo.PushBack("Warp");
	resultsInfo.PushBack("Block64");
	resultsInfo.PushBack("Block128");
	resultsInfo.PushBack("Block256");
	resultsInfo.PushBack("Block512");
	resultsInfo.PushBack("Device8");
	resultsInfo.PushBack("Device16");
	resultsInfo.PushBack("Device32");
	
	Json::Array resultsRow;
	Json::Array resultsColumn;
	Json::Array resultsBatch;
	for (int N=minN; N<=maxN; N*=2)
	{
		int outerDim1 = int(std::sqrt(float(size / N)));
		int outerDim2 = size / (N * outerDim1);
		std::cout << "Benchmark, N=" << N << ", B=" << outerDim1 << "*" << outerDim2 << std::endl;
		Timings t = {0};

		t.reset();
		benchmark(N, outerDim1, outerDim2, "float", "row", false); //dry run
		for (int i=0; i<runs; ++i)
			t += benchmark(N, outerDim1, outerDim2, "float", "row", false);
		t /= runs;
		resultsRow.PushBack(timingsToArray(N, t));

		t.reset();
		benchmark(outerDim1, N, outerDim2, "float", "col", false);
		for (int i = 0; i < runs; ++i)
			t += benchmark(outerDim1, N, outerDim2, "float", "col", false);
		t /= runs;
		resultsColumn.PushBack(timingsToArray(N, t));

		t.reset();
		benchmark(outerDim1, outerDim2, N, "float", "batch", false);
		for (int i = 0; i < runs; ++i)
			t += benchmark(outerDim1, outerDim2, N, "float", "batch", false);
		t /= runs;
		resultsBatch.PushBack(timingsToArray(N, t));
	}
	results.Insert(std::make_pair("Row", resultsRow));
	results.Insert(std::make_pair("Column", resultsColumn));
	results.Insert(std::make_pair("Batch", resultsBatch));

	std::cout << "Make plots" << std::endl;
	std::ofstream outStream(outputDir + "batched_reductions_" + std::to_string(sizePower) + ".json");
	outStream << results;
	outStream.close();

	std::string launchParams = "\"" + pythonPath + " " 
		+ std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlots.py"
		+ " \"" + outputDir + "batched_reductions_" + std::to_string(sizePower) + "\" "
		+ "\"";
	std::cout << launchParams << std::endl;
	system(launchParams.c_str());

    std::cout << "DONE" << std::endl;
}
