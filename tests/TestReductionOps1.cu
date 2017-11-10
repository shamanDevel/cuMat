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

TEST_CASE("raw_reduce_none", "[reduce]")
{
    //TODO
}

TEST_CASE("raw_reduce_R", "[reduce]")
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

TEST_CASE("raw_reduce_C", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 2);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, ReductionAxis::Column, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(8);
    out.copyToHost(&result[0]);
    REQUIRE(result[0] == 6);
    REQUIRE(result[1] == 15);
    REQUIRE(result[2] == 24);
    REQUIRE(result[3] == 33);
    REQUIRE(result[4] == 42);
    REQUIRE(result[5] == 51);
    REQUIRE(result[6] == 60);
    REQUIRE(result[7] == 69);
}

TEST_CASE("raw_reduce_B", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 3, 1);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, ReductionAxis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(12);
    out.copyToHost(&result[0]);
    REQUIRE(result[0] == 14);
    REQUIRE(result[1] == 16);
    REQUIRE(result[2] == 18);
    REQUIRE(result[3] == 20);
    REQUIRE(result[4] == 22);
    REQUIRE(result[5] == 24);
    REQUIRE(result[6] == 26);
    REQUIRE(result[7] == 28);
    REQUIRE(result[8] == 30);
    REQUIRE(result[9] == 32);
    REQUIRE(result[10] == 34);
    REQUIRE(result[11] == 36);
}

TEST_CASE("raw_reduce_RC", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 1, 2);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, ReductionAxis::Row | ReductionAxis::Column, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(2);
    out.copyToHost(&result[0]);
    REQUIRE(result[0] == 78);
    REQUIRE(result[1] == 222);
}

TEST_CASE("raw_reduce_RB", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 1);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, ReductionAxis::Row | ReductionAxis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(3);
    out.copyToHost(&result[0]);
    REQUIRE(result[0] == 92);
    REQUIRE(result[1] == 100);
    REQUIRE(result[2] == 108);
}

TEST_CASE("raw_reduce_CB", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 1);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, ReductionAxis::Column | ReductionAxis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(4);
    out.copyToHost(&result[0]);
    REQUIRE(result[0] == 48);
    REQUIRE(result[1] == 66);
    REQUIRE(result[2] == 84);
    REQUIRE(result[3] == 102);
}

TEST_CASE("raw_reduce_RCB", "[reduce]")
{
    auto m = createTestMatrix();
    Scalari out;
    internal::ReductionEvaluator<BMatrixXiR, Scalari, ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    int result;
    out.copyToHost(&result);
    REQUIRE(result == 300);
}