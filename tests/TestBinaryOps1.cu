#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestBinaryOps.cuh"

using namespace cuMat::functions;

BINARY_TEST_CASE_ALL(add, X + Y, X + Y, -1000, 1000)
BINARY_TEST_CASE_ALL(sub, X - Y, X - Y, -1000, 1000)
//BINARY_TEST_CASE_INT(modulo, X % Y, X % Y, -1000, 1000); //no modulo in Eigen
BINARY_TEST_CASE_FLOAT(power, pow(X,Y), pow(X.array(),Y.array()), 0.01, 10)
BINARY_TEST_CASE_ALL(mul, X.cwiseMul(Y), X.array()*Y.array(), -1000, 1000)
BINARY_TEST_CASE_ALL(div, X.cwiseDiv(Y), X.array()/Y.array(), 0.01, 1000)
BINARY_TEST_CASE_ALL(divInv, X.cwiseDivInv(Y), Y.array() / X.array(), 0.01, 1000)
