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

TEST_CASE("raw_reduce_none_sum", "[reduce]")
{
	auto m = createTestMatrix();
	BMatrixXiR out(4, 3, 2);
	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Block512") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Block<512>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	assertMatrixEquality(m, out);
}

TEST_CASE("raw_reduce_RCB_sum", "[reduce]")
{
	auto m = createTestMatrix();
	Scalari out;
	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Block512") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Block<512>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, Scalari, Axis::All, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "full");
	}
	int result;
	out.copyToHost(&result);
	REQUIRE(result == 300);
}

TEST_CASE("raw_reduce_R_sum", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 2);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}

    std::vector<int> result(6);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 22);
    CHECK(result[1] == 26);
    CHECK(result[2] == 30);
    CHECK(result[3] == 70);
    CHECK(result[4] == 74);
    CHECK(result[5] == 78);
}

TEST_CASE("raw_reduce_C_sum", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 2);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}

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

TEST_CASE("raw_reduce_B_sum", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 3, 1);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}

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

TEST_CASE("raw_reduce_RC_sum", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 1, 2);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
    //internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Column, functor::Sum<int>, int>::eval(m, out, functor::Sum<int>(), 0);

    std::vector<int> result(2);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 78);
    CHECK(result[1] == 222);
}

TEST_CASE("raw_reduce_RB_sum", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 1);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
    //internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Sum<int>, int>::eval(m, out, functor::Sum<int>(), 0);

    std::vector<int> result(3);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 92);
    CHECK(result[1] == 100);
    CHECK(result[2] == 108);
}

TEST_CASE("raw_reduce_CB_sum", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 1);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Sum<int>(), 0);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
    //internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Sum<int>, int>::eval(m, out, functor::Sum<int>(), 0);

    std::vector<int> result(4);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 48);
    CHECK(result[1] == 66);
    CHECK(result[2] == 84);
    CHECK(result[3] == 102);
}
