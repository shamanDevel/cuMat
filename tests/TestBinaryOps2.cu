#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestBinaryOps.cuh"

// Test for broadcasting

using namespace cuMat;

TEST_CASE("broadcasting-multiply", "[binary]")
{
    int data[2][3][4] = {
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

    BMatrixXiR m1 = BMatrixXiR::fromArray(data);
    BMatrixXiC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    SECTION("host scalar")
    {
        int expected[2][3][4] = {
            {
                { 10, 4, 8, 2 },
                { 16, 10, 12, 4 },
                { -10,0, 6,-18 }
            },
            {
                { 22,0, 0, 6 },
                { -14,-4,-8,2 },
                { 18, 8, 14,-14 }
            }
        };
        assertMatrixEquality(expected, 2 * m1);
        assertMatrixEquality(expected, 2 * m2);
        assertMatrixEquality(expected, m1 * 2);
        assertMatrixEquality(expected, m2 * 2);
    }

    SECTION("device scalar")
    {
        int expected[2][3][4] = {
            {
                { 10, 4, 8, 2 },
                { 16, 10, 12, 4 },
                { -10,0, 6,-18 }
            },
            {
                { 22,0, 0, 6 },
                { -14,-4,-8,2 },
                { 18, 8, 14,-14 }
            }
        };
        int scalar[1][1][1] = { {{2}} };
        ScalariR s1 = ScalariR::fromArray(scalar);
        ScalariC s2 = s1.block<1, 1, 1>(0, 0, 0);
        //note: no operator to avoid ambiguity with matrix product
        assertMatrixEquality(expected, s1.cwiseMul(m1));
        assertMatrixEquality(expected, s1.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s1));
        assertMatrixEquality(expected, m2.cwiseMul(s1));
        assertMatrixEquality(expected, s2.cwiseMul(m1));
        assertMatrixEquality(expected, s2.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s2));
        assertMatrixEquality(expected, m2.cwiseMul(s2));
    }

    SECTION("device column")
    {
        int expected[2][3][4] = {
            {
                { 10, 4, 8, 2 },
                { -8, -5, -6, -2 },
                { -5,0, 3,-9 }
            },
            {
                { 22,0, 0, 6 },
                { 7,2,4,-1 },
                { 9, 4, 7,-7 }
            }
        };
        int other[1][3][1] = { { { 2 }, {-1}, {1} } };
        VectorXiR s1 = VectorXiR::fromArray(other);
        VectorXiR s2 = s1.block<Dynamic, 1, 1>(0, 0, 0, s1.rows(), s1.cols(), s1.batches());
        //note: no operator to avoid ambiguity with matrix product
        assertMatrixEquality(expected, s1.cwiseMul(m1));
        assertMatrixEquality(expected, s1.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s1));
        assertMatrixEquality(expected, m2.cwiseMul(s1));
        assertMatrixEquality(expected, s2.cwiseMul(m1));
        assertMatrixEquality(expected, s2.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s2));
        assertMatrixEquality(expected, m2.cwiseMul(s2));
    }

    SECTION("device row")
    {
        int expected[2][3][4] = {
            {
                { 10, 2, 0, -2 },
                { 16, 5, 0, -4 },
                { -10,0, 0,18 }
            },
            {
                { 22,0, 0, -6 },
                { -14,-2,0,-2 },
                { 18, 4, 0,14 }
            }
        };
        int other[1][1][4] = { { { 2, 1, 0, -2 } } };
        RowVectorXiR s1 = RowVectorXiR::fromArray(other);
        RowVectorXiR s2 = s1.block<1, Dynamic, 1>(0, 0, 0, s1.rows(), s1.cols(), s1.batches());
        //note: no operator to avoid ambiguity with matrix product
        assertMatrixEquality(expected, s1.cwiseMul(m1));
        assertMatrixEquality(expected, s1.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s1));
        assertMatrixEquality(expected, m2.cwiseMul(s1));
        assertMatrixEquality(expected, s2.cwiseMul(m1));
        assertMatrixEquality(expected, s2.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s2));
        assertMatrixEquality(expected, m2.cwiseMul(s2));
    }

    SECTION("device batch")
    {
        int expected[2][3][4] = {
            {
                { 10, 4, 8, 2 },
                { 16, 10, 12, 4 },
                { -10,0, 6,-18 }
            },
            {
                { -22,0, 0, -6 },
                { 14, 4,8,-2 },
                { -18, -8, -14, 14 }
            }
        };
        int scalar[2][1][1] = { { { 2 } }, {{-2}} };
        BScalariR s1 = BScalariR::fromArray(scalar);
        BScalariC s2 = s1.block<1, 1, Dynamic>(0, 0, 0, s1.rows(), s1.cols(), s1.batches());
        //note: no operator to avoid ambiguity with matrix product
        assertMatrixEquality(expected, s1.cwiseMul(m1));
        assertMatrixEquality(expected, s1.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s1));
        assertMatrixEquality(expected, m2.cwiseMul(s1));
        assertMatrixEquality(expected, s2.cwiseMul(m1));
        assertMatrixEquality(expected, s2.cwiseMul(m2));
        assertMatrixEquality(expected, m1.cwiseMul(s2));
        assertMatrixEquality(expected, m2.cwiseMul(s2));
    }
}




TEST_CASE("broadcasting-substract", "[binary]")
{
    int data[2][3][4] = {
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

    BMatrixXiR m1 = BMatrixXiR::fromArray(data);
    BMatrixXiC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    SECTION("host scalar")
    {
        int expectedLeft[2][3][4] = {
            {
                { 5, 8, 6, 9 },
                { 2, 5, 4, 8 },
                { 15,10, 7,19 }
            },
            {
                { -1,10, 10, 7 },
                { 17,12,14,9 },
                { 1, 6, 3,17 }
            }
        };
        int expectedRight[2][3][4] = {
            {
                { -5, -8, -6, -9 },
                { -2, -5, -4, -8 },
                { -15, -10, -7,-19 }
            },
            {
                { 1,-10, -10, -7 },
                { -17,-12,-14,-9 },
                { -1, -6, -3,-17 }
            }
        };
        assertMatrixEquality(expectedLeft, 10 - m1);
        assertMatrixEquality(expectedLeft, 10 - m2);
        assertMatrixEquality(expectedRight, m1 - 10);
        assertMatrixEquality(expectedRight, m2 - 10);
    }

    SECTION("device scalar")
    {
        int expectedLeft[2][3][4] = {
            {
                { 5, 8, 6, 9 },
                { 2, 5, 4, 8 },
                { 15,10, 7,19 }
            },
            {
                { -1,10, 10, 7 },
                { 17,12,14,9 },
                { 1, 6, 3,17 }
            }
        };
        int expectedRight[2][3][4] = {
            {
                { -5, -8, -6, -9 },
                { -2, -5, -4, -8 },
                { -15, -10, -7,-19 }
            },
            {
                { 1,-10, -10, -7 },
                { -17,-12,-14,-9 },
                { -1, -6, -3,-17 }
            }
        };
        int scalar[1][1][1] = { { { 10 } } };
        ScalariR s1 = ScalariR::fromArray(scalar);
        ScalariC s2 = s1.block<1, 1, 1>(0, 0, 0);
        assertMatrixEquality(expectedLeft, s1 - m1);
        assertMatrixEquality(expectedLeft, s1 - m2);
        assertMatrixEquality(expectedRight, m1 - s1);
        assertMatrixEquality(expectedRight, m2 - s1);
        assertMatrixEquality(expectedLeft, s2 - m1);
        assertMatrixEquality(expectedLeft, s2 - m2);
        assertMatrixEquality(expectedRight, m1 - s2);
        assertMatrixEquality(expectedRight, m2 - s2);
    }

    SECTION("device column")
    {
        int expectedLeft[2][3][4] = {
            {
                { 5, 8, 6, 9 },
                { -8, -5, -6, -2 },
                { -5,-10, -13,-1 }
            },
            {
                { -1,10, 10, 7 },
                { 7,2,4,-1 },
                { -19, -14, -17,-3 }
            }
        };
        int expectedRight[2][3][4] = {
            {
                { -5, -8, -6, -9 },
                { 8, 5, 6, 2 },
                { 5,10, 13,1 }
            },
            {
                { 1,-10, -10, -7 },
                { -7,-2,-4,1 },
                { 19, 14, 17,3 }
            }
        };
        int other[1][3][1] = { { { 10 },{ 0 },{ -10 } } };
        VectorXiR s1 = VectorXiR::fromArray(other);
        VectorXiR s2 = s1.block<Dynamic, 1, 1>(0, 0, 0, s1.rows(), s1.cols(), s1.batches());
        assertMatrixEquality(expectedLeft, s1 - m1);
        assertMatrixEquality(expectedLeft, s1 - m2);
        assertMatrixEquality(expectedRight, m1 - s1);
        assertMatrixEquality(expectedRight, m2 - s1);
        assertMatrixEquality(expectedLeft, s2 - m1);
        assertMatrixEquality(expectedLeft, s2 - m2);
        assertMatrixEquality(expectedRight, m1 - s2);
        assertMatrixEquality(expectedRight, m2 - s2);
    }

    SECTION("device row")
    {
        int expectedLeft[2][3][4] = {
            {
                { 5, -2, -14, 0 },
                { 2, -5, -16, -1 },
                { 15,0, -13,10 }
            },
            {
                { -1,0, -10, -2 },
                { 17,2,-6,0 },
                { 1, -4, -17,8 }
            }
        };
        int expectedRight[2][3][4] = {
            {
                { -5, 2, 14, 0 },
                { -2, 5, 16, 1 },
                { -15,0, 13,-10 }
            },
            {
                { 1,0, 10, 2 },
                { -17,-2,6,0 },
                { -1, 4, 17,-8 }
            }
        };
        int other[1][1][4] = { { { 10, 0, -10, 1 } } };
        RowVectorXiR s1 = RowVectorXiR::fromArray(other);
        RowVectorXiR s2 = s1.block<1, Dynamic, 1>(0, 0, 0, s1.rows(), s1.cols(), s1.batches());
        assertMatrixEquality(expectedLeft, s1 - m1);
        assertMatrixEquality(expectedLeft, s1 - m2);
        assertMatrixEquality(expectedRight, m1 - s1);
        assertMatrixEquality(expectedRight, m2 - s1);
        assertMatrixEquality(expectedLeft, s2 - m1);
        assertMatrixEquality(expectedLeft, s2 - m2);
        assertMatrixEquality(expectedRight, m1 - s2);
        assertMatrixEquality(expectedRight, m2 - s2);
    }

    SECTION("device batch")
    {
        int expectedLeft[2][3][4] = {
            {
                { 5, 8, 6, 9 },
                { 2, 5, 4, 8 },
                { 15,10, 7,19 }
            },
            {
                { -21,-10, -10, -13 },
                { -3,-8,-6,-11 },
                { -19, -14, -17,-3 }
            }
        };
        int expectedRight[2][3][4] = {
            {
                { -5, -8, -6, -9 },
                { -2, -5, -4, -8 },
                { -15,-10,-7, -19 }
            },
            {
                { 21,10, 10, 13 },
                { 3,8,6,11 },
                { 19, 14, 17,3 }
            }
        };
        int scalar[2][1][1] = { { { 10 } },{ { -10 } } };
        BScalariR s1 = BScalariR::fromArray(scalar);
        BScalariC s2 = s1.block<1, 1, Dynamic>(0, 0, 0, s1.rows(), s1.cols(), s1.batches());
        assertMatrixEquality(expectedLeft, s1 - m1);
        assertMatrixEquality(expectedLeft, s1 - m2);
        assertMatrixEquality(expectedRight, m1 - s1);
        assertMatrixEquality(expectedRight, m2 - s1);
        assertMatrixEquality(expectedLeft, s2 - m1);
        assertMatrixEquality(expectedLeft, s2 - m2);
        assertMatrixEquality(expectedRight, m1 - s2);
        assertMatrixEquality(expectedRight, m2 - s2);
    }
}