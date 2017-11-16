#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>
#include <cuMat/src/ReductionOps.h>

#include "Utils.h"

using namespace cuMat;

// high-level test

TEST_CASE("reduce_sum", "[reduce]")
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

    //row+column+batch
    INFO("reduce: all");
    int expected[1][1][1] = { {{37}} };
    assertMatrixEquality(expected, m1.sum());
    assertMatrixEquality(expected, m1.sum(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
    assertMatrixEquality(expected, m2.sum());
    assertMatrixEquality(expected, m2.sum(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
}

TEST_CASE("reduce_prod", "[reduce]")
{
    int data[2][3][2] = {
        {
            { 1, 2 },
            { 1, 1 },
            { -5,1 }
        },
        {
            { 2, 1 },
            { -3,-2 },
            { 2, 1 }
        }
    };

    cuMat::BMatrixXiR m1 = cuMat::BMatrixXiR::fromArray(data);
    cuMat::BMatrixXiC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    INFO("reduce: all");
    int expected[1][1][1] = { { { -240 } } };
    assertMatrixEquality(expected, m1.prod());
    assertMatrixEquality(expected, m1.prod(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
    assertMatrixEquality(expected, m2.prod());
    assertMatrixEquality(expected, m2.prod(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
}

TEST_CASE("reduce_min", "[reduce]")
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

    cuMat::BMatrixXiR m1 = cuMat::BMatrixXiR::fromArray(data);
    cuMat::BMatrixXiC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    INFO("reduce: all");
    int expected[1][1][1] = { { { -9 } } };
    assertMatrixEquality(expected, m1.minCoeff());
    assertMatrixEquality(expected, m1.minCoeff(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
    assertMatrixEquality(expected, m2.minCoeff());
    assertMatrixEquality(expected, m2.minCoeff(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
}

TEST_CASE("reduce_max", "[reduce]")
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

    cuMat::BMatrixXiR m1 = cuMat::BMatrixXiR::fromArray(data);
    cuMat::BMatrixXiC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    INFO("reduce: all");
    int expected[1][1][1] = { { { 11 } } };
    assertMatrixEquality(expected, m1.maxCoeff());
    assertMatrixEquality(expected, m1.maxCoeff(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
    assertMatrixEquality(expected, m2.maxCoeff());
    assertMatrixEquality(expected, m2.maxCoeff(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch));
}

TEST_CASE("reduce_all", "[reduce]")
{
    bool data[2][3][4] = {
        {
            { true, true, true, true },
            { true, true, true, true },
            { true, true, true, true }
        },
        {
            { true,false, false, true },
            { true,true,true,true },
            { true, true, true,true }
        }
    };

    cuMat::BMatrixXbR m1 = cuMat::BMatrixXbR::fromArray(data);
    cuMat::BMatrixXbC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    INFO("reduce: row+col");
    bool expected[2][1][1] = { { { true } }, {{false}} };
    assertMatrixEquality(expected, m1.all<ReductionAxis::Row | ReductionAxis::Column>());
    assertMatrixEquality(expected, m1.all(ReductionAxis::Row | ReductionAxis::Column));
    assertMatrixEquality(expected, m2.all<ReductionAxis::Row | ReductionAxis::Column>());
    assertMatrixEquality(expected, m2.all(ReductionAxis::Row | ReductionAxis::Column));
}

TEST_CASE("reduce_any", "[reduce]")
{
    bool data[2][3][4] = {
        {
            { false, false, false, false },
            { false, false, false, false },
            { false, false, false, false }
        },
        {
            { true,false, false, true },
            { true,true,true,true },
            { true, true, true,true }
        }
    };

    cuMat::BMatrixXbR m1 = cuMat::BMatrixXbR::fromArray(data);
    cuMat::BMatrixXbC m2 = m1.block(0, 0, 0, m1.rows(), m1.cols(), m1.batches()); //this line forces cwise-evaluation

    INFO("reduce: row+col");
    bool expected[2][1][1] = { { { false } },{ { true} } };
    assertMatrixEquality(expected, m1.any<ReductionAxis::Row | ReductionAxis::Column>());
    assertMatrixEquality(expected, m1.any(ReductionAxis::Row | ReductionAxis::Column));
    assertMatrixEquality(expected, m2.any<ReductionAxis::Row | ReductionAxis::Column>());
    assertMatrixEquality(expected, m2.any(ReductionAxis::Row | ReductionAxis::Column));
}