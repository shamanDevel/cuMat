#include <catch/catch.hpp>

#include <cuMat/Core>

#include "Utils.h"

using namespace cuMat;

TEST_CASE("Compound-Matrix", "[Compound]")
{
    int data[1][2][2] = { {{1, 2}, {3, 4}} };
    Matrix2iR mat1 = Matrix2iR::fromArray(data);
    Matrix2iR mat2;
    
    // ADD
    mat2 = mat1.deepClone();
    mat2 += mat1;
    int expected1[1][2][2] = { { { 2, 4 },{ 6, 8 } } };
    assertMatrixEquality(expected1, mat2);

    // SUB
    mat2 = Matrix2iR::Zero();
    mat2 -= mat1;
    int expected2[1][2][2] = { { { -1, -2 },{ -3, -4 } } };
    assertMatrixEquality(expected2, mat2);

    // MOD
    mat2 = mat1 + 3;
    mat2 %= mat1;
    int expected3[1][2][2] = { { { 0, 1 },{ 0, 3 } } };
    assertMatrixEquality(expected3, mat2);
}