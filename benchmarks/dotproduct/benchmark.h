/*
 * General entry points to benchmarks
 */

#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <vector>
#include <string>
#include "../json_st.h"

/**
 * \brief Launches the implementations of cuMat.
 * Implemented per benchmark
 * \param parameterNames the parameter names
 * \param parameters the parameter values
 * \param returnNames 
 * \param returnValues 
 */
void benchmark_cuMat(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues);

void benchmark_CUB(
	const std::vector<std::string>& parameterNames,
	const Json::Array& parameters,
	const std::vector<std::string>& returnNames,
	Json::Array& returnValues);

void benchmark_Thrust(
	const std::vector<std::string>& parameterNames,
	const Json::Array& parameters,
	const std::vector<std::string>& returnNames,
	Json::Array& returnValues);

void benchmark_cuBlas(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues);

void benchmark_Eigen(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues);

#endif
