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

TEST_CASE("raw_reduce_none_prod", "[reduce]")
{
	auto m = createTestMatrix();
	BMatrixXiR out(4, 3, 2);
	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Block512") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Block<512>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, 0, functor::Prod<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "noop");
	}
	assertMatrixEquality(m, out);
}

TEST_CASE("raw_reduce_R_prod", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 2);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row, functor::Prod<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}

    std::vector<int> result(6);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 280);
    CHECK(result[1] == 880);
    CHECK(result[2] == 1944);
    CHECK(result[3] == 86944);
    CHECK(result[4] == 109480);
    CHECK(result[5] == 136080);
}

TEST_CASE("raw_reduce_C_prod", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 2);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column, functor::Prod<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}

    std::vector<int> result(8);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 6);
    CHECK(result[1] == 120);
    CHECK(result[2] == 504);
    CHECK(result[3] == 1320);
    CHECK(result[4] == 2730);
    CHECK(result[5] == 4896);
    CHECK(result[6] == 7980);
    CHECK(result[7] == 12144);
}

TEST_CASE("raw_reduce_B_prod", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 3, 1);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Batch, functor::Prod<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}

    std::vector<int> result(12);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 13);
    CHECK(result[1] == 28);
    CHECK(result[2] == 45);
    CHECK(result[3] == 64);
    CHECK(result[4] == 85);
    CHECK(result[5] == 108);
    CHECK(result[6] == 133);
    CHECK(result[7] == 160);
    CHECK(result[8] == 189);
    CHECK(result[9] == 220);
    CHECK(result[10] == 253);
    CHECK(result[11] == 288);
}

TEST_CASE("raw_reduce_RB_prod", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(1, 3, 1);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
    //internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Row | Axis::Batch, functor::Prod<int>, int>::eval(m, out, functor::Prod<int>(), 1);

    std::vector<int> result(3);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 24344320);
    CHECK(result[1] == 96342400);
    CHECK(result[2] == 264539520);
}

TEST_CASE("raw_reduce_CB_prod", "[reduce]")
{
    auto m = createTestMatrix();
    BMatrixXiR out(4, 1, 1);

	SECTION("Segmented") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Segmented>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Segmented");
	}
	SECTION("Thread") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Thread>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
	SECTION("Warp") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Warp>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Warp");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Block256") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Block<256>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Block<256>");
	}
	SECTION("Device1") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Device<1>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<1>");
	}
	SECTION("Device2") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Device<2>>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Device<2>");
	}
	SECTION("Auto") {
		internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int, ReductionAlg::Auto>::eval(m, out, functor::Prod<int>(), 1);
		REQUIRE(LastReductionAlgorithm == "Thread");
	}
    //internal::ReductionEvaluator<BMatrixXiR, BMatrixXiR, Axis::Column | Axis::Batch, functor::Prod<int>, int>::eval(m, out, functor::Prod<int>(), 1);
    std::vector<int> result(4);
    out.copyToHost(&result[0]);
    CHECK(result[0] == 16380);
    CHECK(result[1] == 587520);
    CHECK(result[2] == 4021920);
    CHECK(result[3] == 16030080);
}
