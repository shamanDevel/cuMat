#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>
#include <cuMat/src/ReductionOps.h>

#include "Utils.h"

using namespace cuMat;

// test of combined functions

TEST_CASE("trace", "[reduce]")
{
    int data[2][4][4] {
        {
            {5, 2, 4, 1},
            {8, 5, 6, 2},
            {-5,0, 3,-9},
            {3, 7, 2, 1}
        },
        {
            {11,0, 0, 3},
            {-7,-2,-4,1},
            {9, 4, -7,-7},
            {8, 3, 4, -4}
        }
    };

    int expected[2][1][1] {
        {{14}},
        {{-2}}
    };

    BMatrixXiR m1 = BMatrixXiR::fromArray(data);
    assertMatrixEquality(expected, m1.trace());
}

TEST_CASE("dot", "[reduce]")
{
    int data1[2][1][3] = {
        {
            { 1, 2, 6 },
        },
        {
            { 2, 1, 3 },
        }
    };
    int data2[1][1][3] = {
        {
            { 5, -2, 8 },
        }
    };
    int expected[2][1][1] {
        {{49}},
        {{32}}
    };

    BRowVectorXiR m1 = BRowVectorXiR::fromArray(data1);
    RowVectorXiR m2 = RowVectorXiR::fromArray(data2);
    assertMatrixEquality(expected, m1.dot(m2));
    assertMatrixEquality(expected, m2.dot(m1));
}

TEST_CASE("norm", "[reduce]")
{
    double data[2][1][4] = {
        {
            { 5, 2, 4, 1 },
        },
        {
            { 11,0, 0, -3 },
        }
    };
    double expectedSquared[2][1][1] {
        {{46}},
        {{130}}
    };
    double expected[2][1][1]{
        { { 6.782329983125268139064556326626 } },
        { { 11.401754250991379791360490255668 } }
    };

    BMatrixXdR m1 = BMatrixXdR::fromArray(data);
    assertMatrixEquality(expectedSquared, m1.squaredNorm());
    assertMatrixEquality(expected, m1.norm());
}
