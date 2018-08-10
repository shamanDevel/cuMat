#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestUnaryOps.cuh"

using namespace cuMat::functions;

UNARY_TEST_CASE_FLOAT(cwiseSin, sin, -100.0, 100);
UNARY_TEST_CASE_FLOAT(cwiseCos, cos, -100.0, 100);
UNARY_TEST_CASE_FLOAT(cwiseTan, tan, -100.0, 100);
UNARY_TEST_CASE_FLOAT(cwiseAsin, asin, -1.0, 1);
UNARY_TEST_CASE_FLOAT(cwiseAcos, acos, -1.0, 1);
UNARY_TEST_CASE_FLOAT(cwiseAtan, atan, -1.0, 1);
UNARY_TEST_CASE_FLOAT(cwiseSinh, sinh, -10, 10);
UNARY_TEST_CASE_FLOAT(cwiseCosh, cosh, -10, 10);
UNARY_TEST_CASE_FLOAT(cwiseTanh, tanh, -10, 10);
//UNARY_TEST_CASE_FLOAT(cwiseAsinh, asinh, -1.0, 1); //Eigen does not support asinh, acosh, atanh
//UNARY_TEST_CASE_FLOAT(cwiseAcosh, acosh, -1.0, 1);
//UNARY_TEST_CASE_FLOAT(cwiseAtanh, atanh, -1.0, 1);