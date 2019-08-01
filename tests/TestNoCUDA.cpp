#include "TestNoCUDA.h"

#include <catch/catch.hpp>
#include <cuMat/Core>
#include <cuMat/Dense>
#include <cuMat/Sparse>

TEST_CASE("TestNoCUDA", "[NoCUDA]")
{
	//test we are not running in NVCC mode
	REQUIRE_FALSE(CUMAT_NVCC);

	//everything except evaluations should work also in plain cpp

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

	//shallow copies should work
	cuMat::BMatrixXiR m2 = m1;
	//deep copies without transposition should work
	cuMat::BMatrixXiR m3 = m1.deepClone();

	//for computations, we have to delegate to a CUDA source file
	int sum = cudaSumAll(m1);
	REQUIRE(sum == 37);
}