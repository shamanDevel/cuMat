#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestBinaryOps.cuh"

BINARY_TEST_CASE_ALL(add, X + Y, X + Y, -1000, 1000);
BINARY_TEST_CASE_ALL(sub, X - Y, X - Y, -1000, 1000);
//BINARY_TEST_CASE_INT(modulo, X % Y, X % Y, -1000, 1000); //no modulo in Eigen
BINARY_TEST_CASE_FLOAT(power, pow(X,Y), pow(X.array(),Y.array()), 0.01, 10);
