#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestUnaryOps.cuh"

UNARY_TEST_CASE_FLOAT(cwiseExp, exp, -10, 10);
UNARY_TEST_CASE_FLOAT(cwiseLog, log, 0.001, 100);
UNARY_TEST_CASE_FLOAT(cwiseLog1p, log1p, 0.001, 100);
UNARY_TEST_CASE_FLOAT(cwiseLog10, log10, 0.001, 100);

UNARY_TEST_CASE_FLOAT(cwiseSqrt, sqrt, 0.0, 100);
UNARY_TEST_CASE_FLOAT(cwiseRsqrt, rsqrt, 0.0001, 100);
//UNARY_TEST_CASE_FLOAT(cwiseCbrt, pow(1.0/3.0), 0.0, 100);
//UNARY_TEST_CASE_FLOAT(cwiseRcbrt, pow(-1.0 / 3.0), 0.0001, 100);
