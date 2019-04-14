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
#include <set>


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
	a.PushBack(timings.device1);
	a.PushBack(timings.device2);
	a.PushBack(timings.device4);
	a.PushBack(timings.device8);
	a.PushBack(timings.device16);
	a.PushBack(timings.device32);
	return a;
}


std::string pythonPath = "\"C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python36_64/python.exe\"";
std::string outputDir = CUMAT_STR(OUTPUT_DIR);

//correctness validations
void runValidation()
{
	//load json
	Json::Object config = Json::ParseFile(std::string(CUMAT_STR(CONFIG_FILE)));

	//validation
	std::cout << "Validate algorithms" << std::endl;
	Json::Array validation = config["Validation"].AsArray();
	for (int i = 0; i < validation.Size(); ++i)
	{
		Json::Array sizes = validation[i].AsArray();
		int rows = sizes[0].AsInt32();
		int cols = sizes[1].AsInt32();
		int batches = sizes[2].AsInt32();
		benchmark(rows, cols, batches, "int", "row", true);
		benchmark(rows, cols, batches, "int", "col", true);
		benchmark(rows, cols, batches, "int", "batch", true);
	}
}

//1D-benchmark
void runLinearBenchmark()
{
	//benchmarks
	int sizePower = 22;
	int minN = 2;
	int maxN = 1 << (sizePower - 2);
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
	resultsInfo.PushBack("Device1");
	resultsInfo.PushBack("Device2");
	resultsInfo.PushBack("Device4");
	resultsInfo.PushBack("Device8");
	resultsInfo.PushBack("Device16");
	resultsInfo.PushBack("Device32");

	Json::Array resultsRow;
	Json::Array resultsColumn;
	Json::Array resultsBatch;
	for (int N = minN; N <= maxN; N *= 2)
	{
		int outerDim1 = int(std::sqrt(float(size / N)));
		int outerDim2 = size / (N * outerDim1);
		std::cout << "Benchmark, N=" << N << ", B=" << outerDim1 << "*" << outerDim2 << std::endl;
		Timings t = { 0 };

		t.reset();
		benchmark(N, outerDim1, outerDim2, "float", "row", false); //dry run
		for (int i = 0; i < runs; ++i)
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
		+ std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlotsLinear.py"
		+ " \"" + outputDir + "batched_reductions_" + std::to_string(sizePower) + "\" "
		+ "\"";
	std::cout << launchParams << std::endl;
	system(launchParams.c_str());
}

//FULL 2D Benchmark
void runFullBenchmark()
{
	long minNumBatches = 1;
	long maxNumBatches = 1 << 22;
	long minBatchSize = 1;
	long maxBatchSize = 1 << 22;//1 << 15;
	long maxTotalSize = 1 << 22;//1 << 22;
	long minTotalSize = 256;
	double step = 0.25;
	int runs = 5;

	std::ofstream outStreamRow(outputDir + "batched_reductions_full_row.txt");
	std::ofstream outStreamCol(outputDir + "batched_reductions_full_col.txt");
	std::ofstream outStreamBatch(outputDir + "batched_reductions_full_batch.txt");
	outStreamRow << "NumBatches\tBatchSize\tCUB\tThread\tWarp"
		<< "\tBlock64\tBlock128\tBlock256\tBlock512\tBlock1024"
		<< "\tDevice1\tDevice2\tDevice4\tDevice8\tDevice16\tDevice32"
		<< std::endl;
	outStreamCol << "NumBatches\tBatchSize\tCUB\tThread\tWarp"
		<< "\tBlock64\tBlock128\tBlock256\tBlock512\tBlock1024"
		<< "\tDevice1\tDevice2\tDevice4\tDevice8\tDevice16\tDevice32"
		<< std::endl;
	outStreamBatch << "NumBatches\tBatchSize\tCUB\tThread\tWarp"
		<< "\tBlock64\tBlock128\tBlock256\tBlock512\tBlock1024"
		<< "\tDevice1\tDevice2\tDevice4\tDevice8\tDevice16\tDevice32"
		<< std::endl;

	std::set<std::tuple<cuMat::Index, cuMat::Index, cuMat::Index>> set;
	for (double i=std::log2(double(minNumBatches)); i<=std::log2(double(maxNumBatches)); i+=step)
	{
		int numBatches = static_cast<int>(std::round(std::pow(2, i)));
		for (double j=std::log2(double(minBatchSize)); j<=std::log2(double(maxBatchSize)); j+=step)
		{
			cuMat::Index batchSize = static_cast<cuMat::Index>(std::round(std::pow(2, j)));
			cuMat::Index totalSize = numBatches * batchSize;
			cuMat::Index outerDim1 = cuMat::Index(std::sqrt(double(numBatches)));
			cuMat::Index outerDim2 = totalSize / (batchSize * outerDim1);
			totalSize = batchSize * outerDim1 * outerDim2;
			auto tuple = std::make_tuple(batchSize, outerDim1, outerDim2);
			if (set.count(tuple) > 0) continue; //already processed
			set.insert(tuple);
			if (totalSize > maxTotalSize)
			{
				std::cout << "Run numBatches=" << numBatches << ", batchSize=" << batchSize << " -> too large, skip" << std::endl;
				continue;
			}
			if (totalSize < minTotalSize)
			{
				std::cout << "Run numBatches=" << numBatches << ", batchSize=" << batchSize << " -> too small, skip" << std::endl;
				continue;
			}

			std::cout << "Run numBatches=" << numBatches << ", batchSize=" << outerDim1 << "*" << outerDim2 << " (=" << totalSize << ")" << std::endl;

			Timings t = { 0 };

			t.reset();
			benchmark(batchSize, outerDim1, outerDim2, "float", "row", false, true); //dry run
			for (int i = 0; i < runs; ++i)
				t += benchmark(batchSize, outerDim1, outerDim2, "float", "row", false, true);
			t /= runs;
			outStreamRow << numBatches << "\t" << batchSize
				<< "\t" << t.baseline << "\t" << t.thread << "\t" << t.warp
				<< "\t" << t.block64 << "\t" << t.block128 << "\t" << t.block256 << "\t" << t.block512 << "\t" << t.block1024
				<< "\t" << t.device1 << "\t" << t.device2 << "\t" << t.device4 << "\t" << t.device8 << "\t" << t.device16 << "\t" << t.device32
				<< std::endl;

			t.reset();
			benchmark(outerDim1, batchSize, outerDim2, "float", "col", false, true);
			for (int i = 0; i < runs; ++i)
				t += benchmark(outerDim1, batchSize, outerDim2, "float", "col", false, true);
			t /= runs;
			outStreamCol << numBatches << "\t" << batchSize
				<< "\t" << t.baseline << "\t" << t.thread << "\t" << t.warp
				<< "\t" << t.block64 << "\t" << t.block128 << "\t" << t.block256 << "\t" << t.block512 << "\t" << t.block1024
				<< "\t" << t.device1 << "\t" << t.device2 << "\t" << t.device4 << "\t" << t.device8 << "\t" << t.device16 << "\t" << t.device32
				<< std::endl;

			t.reset();
			benchmark(outerDim1, outerDim2, batchSize, "float", "batch", false, true);
			for (int i = 0; i < runs; ++i)
				t += benchmark(outerDim1, outerDim2, batchSize, "float", "batch", false, true);
			t /= runs;
			outStreamBatch << numBatches << "\t" << batchSize
				<< "\t" << t.baseline << "\t" << t.thread << "\t" << t.warp
				<< "\t" << t.block64 << "\t" << t.block128 << "\t" << t.block256 << "\t" << t.block512 << "\t" << t.block1024
				<< "\t" << t.device1 << "\t" << t.device2 << "\t" << t.device4 << "\t" << t.device8 << "\t" << t.device16 << "\t" << t.device32
				<< std::endl;
		}
	}

	outStreamRow.close();
	outStreamCol.close();
	outStreamBatch.close();

	std::string launchParams;
	launchParams = "\"" + pythonPath + " "
		+ std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlotsFull.py"
		+ " \"" + outputDir + "batched_reductions_full_row" + "\""
		+ " Row "
		+ "\"";
	std::cout << launchParams << std::endl;
	system(launchParams.c_str());
	launchParams = "\"" + pythonPath + " "
		+ std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlotsFull.py"
		+ " \"" + outputDir + "batched_reductions_full_col" + "\""
		+ " Column "
		+ "\"";
	std::cout << launchParams << std::endl;
	system(launchParams.c_str());
	launchParams = "\"" + pythonPath + " "
		+ std::string(CUMAT_STR(PYTHON_FILES)) + "MakePlotsFull.py"
		+ " \"" + outputDir + "batched_reductions_full_batch" + "\""
		+ " Batch "
		+ "\"";
	std::cout << launchParams << std::endl;
	system(launchParams.c_str());
}

int main(int argc, char* argv[])
{
	std::cout << "==============================" << std::endl;
	std::cout << " VALIDATION" << std::endl;
	std::cout << "==============================" << std::endl;
	runValidation();

	std::cout << "==============================" << std::endl;
	std::cout << " LINEAR BENCHMARK" << std::endl;
	std::cout << "==============================" << std::endl;
	runLinearBenchmark();

	std::cout << "==============================" << std::endl;
	std::cout << " FULL BENCHMARK" << std::endl;
	std::cout << "==============================" << std::endl;
	runFullBenchmark();

    std::cout << "DONE" << std::endl;
}
