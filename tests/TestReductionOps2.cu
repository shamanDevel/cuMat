#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>
#include <cuMat/src/ReductionOps.h>

#include "Utils.h"

using namespace cuMat;

// high-level test, tests reductions over the axis using 'sum'

TEST_CASE("axis", "[reduce]")
{
    int data[2][3][4] = {
        {
            {5, 2, 4, 1},
            {8, 5, 6, 2},
            {-5,0, 3,-9}
        },
        {
            {11,0, 0, 3},
            {-7,-2,-4,1},
            {9, 4, 7,-7}
        }
    };

    cuMat::BMatrixXiR m1 = cuMat::BMatrixXiR::fromArray(data);
    cuMat::BMatrixXiC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    {//none
        INFO("reduce: none");
        int expected[2][3][4] = {
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
        assertMatrixEquality(expected, m1.sum<0>());
        assertMatrixEquality(expected, m1.sum(0));
        assertMatrixEquality(expected, m2.sum<0>());
        assertMatrixEquality(expected, m2.sum(0));
    }

    {//column
        INFO("reduce: column");
        int expected[2][3][1] = {
            {
                { 12 },
                { 21 },
                { -11 }
            },
            {
                { 14 },
                { -12 },
                { 13 }
            }
        };
        assertMatrixEquality(expected, m1.sum<Axis::Column>());
        assertMatrixEquality(expected, m1.sum(Axis::Column));
        assertMatrixEquality(expected, m2.sum<Axis::Column>());
        assertMatrixEquality(expected, m2.sum(Axis::Column));
    }

    {//row
        INFO("reduce: row");
        int expected[2][1][4] = {
            {
                { 8, 7, 13, -6 }
            },
            {
                { 13, 2, 3, -3 }
            }
        };
        assertMatrixEquality(expected, m1.sum<Axis::Row>());
        assertMatrixEquality(expected, m1.sum(Axis::Row));
        assertMatrixEquality(expected, m2.sum<Axis::Row>());
        assertMatrixEquality(expected, m2.sum(Axis::Row));
    }

    {//batch
        INFO("reduce: batch");
        int expected[1][3][4] = {
            {
                { 16, 2, 4, 4 },
                { 1, 3, 2, 3 },
                { 4,4, 10,-16 }
            }
        };
        assertMatrixEquality(expected, m1.sum<Axis::Batch>());
        assertMatrixEquality(expected, m1.sum(Axis::Batch));
        assertMatrixEquality(expected, m2.sum<Axis::Batch>());
        assertMatrixEquality(expected, m2.sum(Axis::Batch));
    }

    {//row+column
        INFO("reduce: row+column");
        int expected[2][1][1] = {
            {
                { 22 }
            },
            {
                { 15 }
            }
        };
        assertMatrixEquality(expected, m1.sum<Axis::Row | Axis::Column>());
        assertMatrixEquality(expected, m1.sum(Axis::Row | Axis::Column));
        assertMatrixEquality(expected, m2.sum<Axis::Row | Axis::Column>());
        assertMatrixEquality(expected, m2.sum(Axis::Row | Axis::Column));
    }

    {//column+batch
        INFO("reduce: column+batch");
        int expected[1][3][1] = {
            {
                { 26 },
                { 9 },
                { 2 }
            }
        };
        assertMatrixEquality(expected, m1.sum<Axis::Column | Axis::Batch>());
        assertMatrixEquality(expected, m1.sum(Axis::Column | Axis::Batch));
        assertMatrixEquality(expected, m2.sum<Axis::Column | Axis::Batch>());
        assertMatrixEquality(expected, m2.sum(Axis::Column | Axis::Batch));
    }

    {//row
        INFO("reduce: row+batch");
        int expected[1][1][4] = {
            {
                { 21, 9, 16, -9 }
            }
        };
        assertMatrixEquality(expected, m1.sum<Axis::Row | Axis::Batch>());
        assertMatrixEquality(expected, m1.sum(Axis::Row | Axis::Batch));
        assertMatrixEquality(expected, m2.sum<Axis::Row | Axis::Batch>());
        assertMatrixEquality(expected, m2.sum(Axis::Row | Axis::Batch));
    }

    {//row+column+batch
        INFO("reduce: all")
        int expected[1][1][1] = { {{37}} };
        assertMatrixEquality(expected, m1.sum());
        assertMatrixEquality(expected, m1.sum(Axis::Row | Axis::Column | Axis::Batch));
        assertMatrixEquality(expected, m2.sum());
        assertMatrixEquality(expected, m2.sum(Axis::Row | Axis::Column | Axis::Batch));
    }
}