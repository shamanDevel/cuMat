#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

// Tests of the primitive reduction ops

using namespace cuMat;

BMatrixXiR createTestMatrix()
{
    int data[2][4][3] = {
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10,11,12 }
        },
        {
            { 13,14,15 },
            { 16,17,18 },
            { 19,20,21 },
            { 22,23,24 }
        }
    };
    return BMatrixXiR::fromArray(data);
}

TEST_CASE("reduce_none", "[reduce]")
{
    //TODO
}

TEST_CASE("reduce_R", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 2);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, ReductionAxis::Row, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(6);
    out.copyToHost(&result[0]);
    REQUIRE(result[0] == 22);
    REQUIRE(result[1] == 26);
    REQUIRE(result[2] == 30);
    REQUIRE(result[3] == 70);
    REQUIRE(result[4] == 74);
    REQUIRE(result[5] == 78);
}

TEST_CASE("reduce_RCB", "[reduce]")
{
    auto m = createTestMatrix();
    Scalari out;
    internal::ReductionEvaluator<BMatrixXiR, Scalari, ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    int result;
    out.copyToHost(&result);
    REQUIRE(result == 300);
}