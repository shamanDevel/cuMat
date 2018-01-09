#include <catch/catch.hpp>
#include <vector>
//#include <unsupported/Eigen/SpecialFunctions>

#include <cuMat/Core>

#include "Utils.h"
#include "TestUnaryOps.cuh"

//UNARY_TEST_CASE_FLOAT(cwiseErf, erf, -1000.0, 1000); //throws THIS_TYPE_IS_NOT_SUPPORTED
//UNARY_TEST_CASE_FLOAT(cwiseErfc, erfc, -1000.0, 1000);
//UNARY_TEST_CASE_FLOAT(cwiseLgamma, lgamma, 0, 10);
//more special functions are not supported directly in CUDA
//Except the bessel functions are supported, but not supported in Eigen (no way to test them easily)
//TODO: add more special functions
