#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"

#define UNARY_OP_HELPER(cuMatFn, eigenFn, min, max) \
	template<typename _Scalar, int _Rows, int _Cols, int _Batches, int _Flags> \
	void unaryOpHelper_ ## cuMatFn (cuMat::Index rows, cuMat::Index cols, cuMat::Index batches) \
	{ \
		std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value >> m_host(batches); \
		for (cuMat::Index i = 0; i < batches; ++i) { \
			m_host[i].setRandom(rows, cols); \
			m_host[i] = _Scalar((min) + ((max)-(min))/2) + _Scalar(((max)-(min))/2) * m_host[i].array();\
		} \
	 \
		cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device(rows, cols, batches); \
		for (cuMat::Index i = 0; i < batches; ++i) { \
			auto slice = cuMat::Matrix<_Scalar, _Rows, _Cols, 1, cuMat::ColumnMajor>::fromEigen(m_host[i]); \
			m_device.block(0, 0, i, rows, cols, 1) = slice; \
		} \
	 \
		cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device1 = m_device. cuMatFn (); \
	 \
		std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value>> m_host1(batches); \
		for (cuMat::Index i = 0; i<batches; ++i) \
		{ \
			m_host1[i] = m_device1.block(0, 0, i, rows, cols, 1).eval().toEigen(); \
		} \
	 \
		for (cuMat::Index i = 0; i<batches; ++i) \
		{ \
			INFO("Input: " << m_host[i]); \
			auto lhs = m_host[i].array(). eigenFn .matrix().eval(); \
			auto rhs = m_host1[i]; \
			INFO("Expected: " << lhs); \
			INFO("Actual: " << rhs); \
			REQUIRE(lhs.isApprox(rhs, Eigen::NumTraits<_Scalar>::dummy_precision() * rows * cols)); \
		} \
	}

#define UNARY_TEST_CASE_ALL(fn, efn, min, max) \
	UNARY_OP_HELPER(fn, efn, min, max) \
	TEST_CASE(CUMAT_STR(fn), "[unary]") \
	{ \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(int, unaryOpHelper_ ## fn); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(long, unaryOpHelper_ ## fn); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(float, unaryOpHelper_ ## fn); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(double, unaryOpHelper_ ## fn); \
	}

#define UNARY_TEST_CASE_FLOAT(fn, efn, min, max) \
	UNARY_OP_HELPER(fn, efn, min, max) \
	TEST_CASE(CUMAT_STR(fn), "[unary]") \
	{ \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(float, unaryOpHelper_ ## fn); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(double, unaryOpHelper_ ## fn); \
	}
#define UNARY_TEST_CASE_FLOAT_NODEF(fn, efn, min, max, name) \
	TEST_CASE(CUMAT_STR(fn) name, "[unary]") \
	{ \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(float, unaryOpHelper_ ## fn); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(double, unaryOpHelper_ ## fn); \
	}
