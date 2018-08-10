#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestUnaryOps.cuh"

template<typename _Scalar, int _Rows, int _Cols, int _Batches, int _Flags>
void TestNegate(cuMat::Index rows, cuMat::Index cols, cuMat::Index batches)
{
	std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value >> m_host(batches);
	for (cuMat::Index i = 0; i < batches; ++i) m_host[i].setRandom(rows, cols);

	cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device(rows, cols, batches);
	for (cuMat::Index i = 0; i < batches; ++i) {
		auto slice = cuMat::Matrix<_Scalar, _Rows, _Cols, 1, cuMat::ColumnMajor>::fromEigen(m_host[i]);
		m_device.block(0, 0, i, rows, cols, 1) = slice;
	}

	cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device1 = m_device.cwiseNegate();
	cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device2 = -m_device;

	std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value>> m_host1(batches);
	std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value>> m_host2(batches);
	for (cuMat::Index i = 0; i<batches; ++i)
	{
		m_host1[i] = m_device1.block(0, 0, i, rows, cols, 1).eval().toEigen();
		m_host2[i] = m_device2.block(0, 0, i, rows, cols, 1).eval().toEigen();
	}

	for (cuMat::Index i = 0; i<batches; ++i)
	{
		REQUIRE(m_host[i] == -m_host1[i]);
		REQUIRE(m_host[i] == -m_host2[i]);
	}
}
TEST_CASE("cwiseNegate", "[unary]")
{
	CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(int, TestNegate);
	CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(long, TestNegate);
	CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(float, TestNegate);
	CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(double, TestNegate);
}



UNARY_TEST_CASE_ALL(cwiseAbs, abs, -10, 10);
UNARY_TEST_CASE_FLOAT(cwiseInverse, inverse, -10, 10);
UNARY_TEST_CASE_ALL(conjugate, conjugate, -10, 10);

UNARY_TEST_CASE_ALL(cwiseFloor, floor, -1000, 1000);
UNARY_TEST_CASE_ALL(cwiseCeil, ceil, -1000, 1000);
UNARY_TEST_CASE_ALL(cwiseRound, round, -1000, 1000);