#include <catch/catch.hpp>
#include <vector>

#define CUMAT_UNITTESTS_LAST_REDUCTION 1
namespace cuMat
{
	std::string LastReductionAlgorithm;
}

#include <cuMat/Core>
#include "Utils.h"

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
	auto m = createTestMatrix();
	SECTION("Segmented") {
		BMatrixXiR out(4, 3, 2);
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, cub::Sum, int, ReductionAlg::Segmented>
			::eval(m, out, cub::Sum(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
		assertMatrixEquality(m, out);
	}
}

#if 0

TEST_CASE("raw_reduce_R", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 2);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(6);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 22);
    CHECK(result[1] == 26);
    CHECK(result[2] == 30);
    CHECK(result[3] == 70);
    CHECK(result[4] == 74);
    CHECK(result[5] == 78);
}

TEST_CASE("raw_reduce_C", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 2);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(8);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 6);
    CHECK(result[1] == 15);
    CHECK(result[2] == 24);
    CHECK(result[3] == 33);
    CHECK(result[4] == 42);
    CHECK(result[5] == 51);
    CHECK(result[6] == 60);
    CHECK(result[7] == 69);
}

TEST_CASE("raw_reduce_B", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 3, 1);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(12);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 14);
    CHECK(result[1] == 16);
    CHECK(result[2] == 18);
    CHECK(result[3] == 20);
    CHECK(result[4] == 22);
    CHECK(result[5] == 24);
    CHECK(result[6] == 26);
    CHECK(result[7] == 28);
    CHECK(result[8] == 30);
    CHECK(result[9] == 32);
    CHECK(result[10] == 34);
    CHECK(result[11] == 36);
}

TEST_CASE("raw_reduce_RC", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 1, 2);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(2);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 78);
    CHECK(result[1] == 222);
}

TEST_CASE("raw_reduce_RB", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 1);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(3);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 92);
    CHECK(result[1] == 100);
    CHECK(result[2] == 108);
}

TEST_CASE("raw_reduce_CB", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 1);
    internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    std::vector<int> result(4);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 48);
    CHECK(result[1] == 66);
    CHECK(result[2] == 84);
    CHECK(result[3] == 102);
}

TEST_CASE("raw_reduce_RCB", "[reduce]")
{
    auto m = createTestMatrix();
    Scalari out;
    internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::Row | Axis::Column | Axis::Batch, cub::Sum, int>
        ::eval(m, out, cub::Sum(), 0);
    int result;
    out.copyToHost(&result);
    CHECK(result == 300);
}

#endif