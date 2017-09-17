#include <catch/catch.hpp>

#include <cuMat/Core>

#include "Utils.h"

template<typename _Source, typename _Target>
void CastingTest()
{
	int rows = 5;
	int cols = 6;
	Eigen::Matrix<_Source, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> in_host;
	in_host.setRandom(rows, cols);
	
	cuMat::Matrix<_Source, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> in_device
		= cuMat::Matrix<_Source, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor>::fromEigen(in_host);

	cuMat::Matrix<_Target, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> out_device
		= in_device.template cast<_Target>();

	Eigen::Matrix<_Target, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_host1
		= out_device.toEigen();
	Eigen::Matrix<_Target, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_host2
		= in_host.template cast<_Target>();

	INFO("Input: " << in_host);
	INFO("Expected: " << out_host2);
	INFO("Actual: " << out_host1);
	REQUIRE(out_host2.isApprox(out_host1, Eigen::NumTraits<_Target>::dummy_precision() * rows * cols));
}
TEST_CASE("casting", "[unary]")
{
	CastingTest<int, int>();
	CastingTest<int, long>();
	CastingTest<int, float>();
	CastingTest<int, double>();

	CastingTest<long, int>();
	CastingTest<long, long>();
	CastingTest<long, float>();
	CastingTest<long, double>();

	CastingTest<float, int>();
	CastingTest<float, long>();
	CastingTest<float, float>();
	CastingTest<float, double>();

	CastingTest<double, int>();
	CastingTest<double, long>();
	CastingTest<double, float>();
	CastingTest<double, double>();
}