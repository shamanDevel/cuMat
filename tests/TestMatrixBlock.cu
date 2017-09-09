#include <catch/catch.hpp>

#include <cuMat/src/Matrix.h>
#include <cuMat/src/MatrixBlock.h>
#include <cuMat/src/EigenInteropHelpers.h>

TEST_CASE("fixed_block_rw", "[block]")
{
	const int rows = 10;
	const int cols = 16;
	const int batches = 8;

	Eigen::Matrix<int, rows, cols, Eigen::ColMajor> m_host1[batches];
	for (int i = 0; i < batches; ++i) m_host1[i].setRandom(rows, cols);

	cuMat::Matrix<int, rows, cols, batches, cuMat::ColumnMajor> m_device;
	for (int i = 0; i < batches; ++i) {
		auto slice = cuMat::Matrix<int, rows, cols, 1, cuMat::ColumnMajor>::fromEigen(m_host1[i]);
		m_device.block<rows, cols, 1>(0, 0, i) = slice;
	}

	Eigen::Matrix<int, rows, cols, Eigen::ColMajor> m_host2[batches];
	for (int i=0; i<batches; ++i)
	{
		m_host2[i] = m_device.block<rows, cols, 1>(0, 0, i).eval().toEigen();
	}

	const cuMat::Matrix<int, rows, cols, batches, cuMat::ColumnMajor> m_device_const = m_device;
	Eigen::Matrix<int, rows, cols, Eigen::ColMajor> m_host3[batches];
	for (int i = 0; i<batches; ++i)
	{
		m_host3[i] = m_device_const.block<rows, cols, 1>(0, 0, i).eval().toEigen();
	}

	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host2[i]);
	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host3[i]);
}

TEST_CASE("dynamic_block_rw", "[block]")
{
	const int rows = 10;
	const int cols = 16;
	const int batches = 8;

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host1[batches];
	for (int i = 0; i < batches; ++i) m_host1[i].setRandom(rows, cols);

	cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> m_device(rows, cols, batches);
	for (int i = 0; i < batches; ++i) {
		auto slice = cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::ColumnMajor>::fromEigen(m_host1[i]);
		m_device.block(0, 0, i, rows, cols, 1) = slice;
	}

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host2[batches];
	for (int i = 0; i<batches; ++i)
	{
		m_host2[i] = m_device.block(0, 0, i, rows, cols, 1).eval().toEigen();
	}

	const cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> m_device_const = m_device;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host3[batches];
	for (int i = 0; i<batches; ++i)
	{
		m_host3[i] = m_device_const.block(0, 0, i, rows, cols, 1).eval().toEigen();
	}

	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host2[i]);
	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host3[i]);
}