#include <catch/catch.hpp>

#include <cuMat/Core>

#include "Utils.h"

using namespace cuMat;

TEST_CASE("Compound-Matrix", "[Compound]")
{
    int data[1][2][2] = { {{1, 2}, {3, 4}} };
    Matrix2iR mat1 = Matrix2iR::fromArray(data);
    Matrix2iR mat2;
    
    SECTION("add") {
        mat2 = mat1.deepClone();
        mat2 += mat1;
        int expected1[1][2][2] = { { { 2, 4 },{ 6, 8 } } };
        assertMatrixEquality(expected1, mat2);
    }

    SECTION("sub") {
        mat2 = Matrix2iR::Zero();
        mat2 -= mat1;
        int expected2[1][2][2] = { { { -1, -2 },{ -3, -4 } } };
        assertMatrixEquality(expected2, mat2);
    }

    SECTION("mod") {
        mat2 = mat1 + 3;
        mat2 %= mat1;
        int expected3[1][2][2] = { { { 0, 1 },{ 0, 3 } } };
        assertMatrixEquality(expected3, mat2);
    }

    SECTION("mul - scalar")
    {
        mat2 = mat1.deepClone();
        mat2 *= 2;
        int expected4[1][2][2] = { { { 2, 4 },{ 6, 8 } } };
        assertMatrixEquality(expected4, mat2);
    }
}

TEST_CASE("Inplace Matmul", "[Compound]")
{
    float rhsData[1][2][2] = { { { 1, 2 },{ 3, 4 } } };
    Matrix2f rhs = Matrix2fR::fromArray(rhsData).deepClone<ColumnMajor>();

    SECTION("vector")
    {
        float lhsData[1][1][2] = { {{5, 4}} };
        RowVector2fR lhs = RowVector2fR::fromArray(lhsData);
        lhs *= rhs;
        float expected[1][1][2] = { {{17, 26}} };
        assertMatrixEquality(expected, lhs);
    }

    SECTION("wrong dimensions")
    {
        MatrixXf lhs(2, 3);
        REQUIRE_THROWS_AS(lhs *= rhs, std::invalid_argument);
    }

    SECTION("matrix")
    {
        float lhsData[1][3][2] = { {{-1, 5}, {2, 8}, {6, -3}} };
        MatrixXf lhs = MatrixXfR::fromArray(lhsData).deepClone<ColumnMajor>();
        lhs *= rhs;
        float expected[1][3][2] = { { { 14, 18 },{ 26, 36 },{ -3, 0 } } };
        assertMatrixEquality(expected, lhs);
    }
}

TEST_CASE("Inplace Matmul result", "[Compound]")
{
    float AData[1][2][2] = { { { 1, 2 },{ 3, 4 } } };
    Matrix2f A = Matrix2fR::fromArray(AData).deepClone<ColumnMajor>();
    float BData[1][2][2] = { { { -3, 7 },{ -2, 8 } } };
    Matrix2f B = Matrix2fR::fromArray(BData).deepClone<ColumnMajor>();

    SECTION("assign")
    {
        Matrix2f C = A * B;
        float expected[1][2][2] = { { { -7, 23 },{ -17, 53 } } };
        assertMatrixEquality(expected, C);
    }

    SECTION("add")
    {
        float lhsData[1][2][2] = { { { -1, 3 }, { 2, -4 } } };
        Matrix2f lhs = Matrix2f::fromArray(lhsData).deepClone<ColumnMajor>();
        lhs += A * B;
        float expected[1][2][2] = { { { -8, 26 },{ -15, 49 } } };
        assertMatrixEquality(expected, lhs);
    }
}
