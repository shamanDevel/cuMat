#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>
#include <cuMat/src/ReductionOps.h>

#include "Utils.h"

using namespace cuMat;

// big reduction test

TEST_CASE("reduce_large", "[reduce]")
{
	static const int sizes[] = { 10 ,100, 1000, 10000, 100000, 1000000 };
	for (int size : sizes)
	{
		Eigen::VectorXd hv = Eigen::VectorXd::Random(size);
		cuMat::VectorXd dv = cuMat::VectorXd::fromEigen(hv);

		double normH = hv.squaredNorm();
		double normD1 = static_cast<double>(dv.squaredNorm());
		double normD2 = static_cast<double>(dv.squaredNorm());
		REQUIRE(normH == Approx(normD1));
		REQUIRE(normH == Approx(normD2));
	}
}