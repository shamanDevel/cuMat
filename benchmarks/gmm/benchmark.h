/*
 * General entry points to benchmarks
 */

#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <vector>
#include <string>
#include "../json_st.h"

void benchmark_cuMat(
	const std::string& pointsFile,
	const std::string& settingsFile,
	int numIterations,
	Json::Object& returnValues);
    
void benchmark_Eigen(
    const std::string& pointsFile,
	const std::string& settingsFile,
	int numIterations,
    Json::Object& returnValues);

#endif
