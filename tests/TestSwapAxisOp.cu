#include <catch/catch.hpp>

#include <cuMat/Core>
#include "Utils.h"

using namespace cuMat;

// high-level test

TEST_CASE("swapAxis1", "[unary]")
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

	BMatrixXiR m1 = BMatrixXiR::fromArray(data);

	BMatrixXiR expected = m1.transpose();
	BMatrixXiR actual = m1.swapAxis<Column, Row, Batch>();
	assertMatrixEquality(expected, actual);
}

TEST_CASE("swapAxis2", "[unary]")
{
	int data[2][3][1] = {
		{
			{5},
			{8},
			{-5}
		},
		{
			{11},
			{-7},
			{9}
		}
	};

	BVectorXi m1 = BVectorXi::fromArray(data);

	int expected[1][3][2] = {
		{
			{5 , 11},
			{8 , -7},
			{-5, 9 }
		}
	};
	MatrixXi actual = m1.swapAxis<Row, Batch, NoAxis>();
	assertMatrixEquality(expected, actual);
}

TEST_CASE("swapAxis3", "[unary]")
{
	int data[1][3][1] = {
		{
			{5},
			{8},
			{-5}
		},
	};

	VectorXi m1 = VectorXi::fromArray(data);

	int expected[3][1][1] = {
		{{5}},
		{{8}},
		{{-5}}
	};
	BScalari actual = m1.swapAxis<NoAxis, NoAxis, Row>();
	assertMatrixEquality(expected, actual);
}