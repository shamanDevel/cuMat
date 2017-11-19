#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestBinaryOps.cuh"

// Test for logic ops

using namespace cuMat;

TEST_CASE("comparation", "[binary]")
{
    float data1[2][3][4] = {
        {
            { 5, 2, 4, 1 },
            { 8, 5, 6, 2 },
            { -5,0, 3,-9 }
        },
        {
            { 11,0, 0, 3 },
            { -7,-2,-4,1 },
            { 9, 4, 7,-7 }
        }
    };
    float data2[2][3][4] = {
        {
            { 4, 3, 4.5, 1.5 },
            { 8, 4, 6, 2 },
            { -6,0.1, 3,-10 }
        },
        {
            { 12,0, 0, 2.5 },
            { -5,-2,-5,-1 },
            { -9, 4, 7,-7 }
        }
    };

    BMatrixXfR m1 = BMatrixXfR::fromArray(data1);
    BMatrixXfR m2 = BMatrixXfR::fromArray(data2);

    SECTION("equal")
    {
        bool expected[2][3][4] = {
            {
                { false, false, false, false },
                { true, false, true, true },
                { false, false, true, false }
            },
            {
                { false, true, true, false },
                { false, true, false, false },
                { false, true, true, true }
            }
        };
        assertMatrixEquality(expected, m1 == m2);
    }

    SECTION("not-equal")
    {
        bool expected[2][3][4] = {
            {
                { true, true, true, true },
                { false, true, false, false },
                { true, true, false, true }
            },
            {
                { true, false, false, true },
                { true, false, true, true },
                { true, false, false, false }
            }
        };
        assertMatrixEquality(expected, m1 != m2);
    }

    SECTION("less")
    {
        bool expected[2][3][4] = {
            {
                { false, true, true, true },
                { false, false, false, false },
                { false, true, false, false }
            },
            {
                { true, false, false, false },
                { true, false, false, false },
                { false, false, false, false }
            }
        };
        assertMatrixEquality(expected, m1 < m2);
    }

    SECTION("greater")
    {
        bool expected[2][3][4] = {
            {
                { true, false, false, false },
                { false, true, false, false },
                { true, false, false, true }
            },
            {
                { false, false, false, true },
                { false, false, true, true },
                { true, false, false, false }
            }
        };
        assertMatrixEquality(expected, m1 > m2);
    }

    SECTION("greater-equals")
    {
        bool expected[2][3][4] = {
            {
                { true, false, false, false},
                { true, true, true, true },
                { true, false, true, true }
            },
            {
                { false, true, true, true },
                { false, true, true, true },
                { true, true, true, true }
            }
        };
        assertMatrixEquality(expected, m1 >= m2);
    }

    SECTION("less-equals")
    {
        bool expected[2][3][4] = {
            {
                { false, true, true, true },
                {true, false, true, true},
                {false, true, true, false}
            },
            {
                {true, true, true, false},
                {true, true, false, false},
                {false, true, true, true}
            }
        };
        assertMatrixEquality(expected, m1 <= m2);
    }
}