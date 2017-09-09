#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"

template<typename _Scalar, int _Rows, int _Cols, int _Batches, int _Flags>
void TestNegate(Index rows, Index cols, Index batches)
{
	std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value >> m_host(batches);
	for (Index i = 0; i < batches; ++i) m_host[i].setRandom(rows, cols);

	cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device(rows, cols, batches);
	for (Index i = 0; i < batches; ++i) {
		auto slice = cuMat::Matrix<_Scalar, _Rows, _Cols, 1, cuMat::ColumnMajor>::fromEigen(m_host[i]);
		m_device.block(0, 0, i, rows, cols, 1) = slice;
	}

	cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device1 = m_device.negate();

	std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value>> m_host1(batches);
	for (Index i = 0; i<batches; ++i)
	{
		m_host1[i] = m_device1.block(0, 0, i, rows, cols, 1).eval().toEigen();
	}

	for (Index i = 0; i<batches; ++i)
	{
		REQUIRE(m_host[i] == -m_host1[i]);
	}
}
TEST_CASE("negate", "[unary]")
{
	CUMAT_TESTS_CALL_MATRIX_TEST(int, TestNegate);
	CUMAT_TESTS_CALL_MATRIX_TEST(long, TestNegate);
	CUMAT_TESTS_CALL_MATRIX_TEST(float, TestNegate);
	CUMAT_TESTS_CALL_MATRIX_TEST(double, TestNegate);
}