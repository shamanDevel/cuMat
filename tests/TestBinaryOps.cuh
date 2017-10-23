#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"

#define BINARY_OP_HELPER(name, cuMatFn, eigenFn, min, max) \
	template<typename _Scalar, int _Rows, int _Cols, int _Batches, int _Flags> \
	void binaryOpHelper_ ## name (cuMat::Index rows, cuMat::Index cols, cuMat::Index batches) \
	{ \
		std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value >> m_host1(batches); \
        std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value >> m_host2(batches); \
		for (cuMat::Index i = 0; i < batches; ++i) { \
			m_host1[i].setRandom(rows, cols); \
			m_host1[i] = _Scalar((min) + ((max)-(min))/2) + _Scalar(((max)-(min))/2) * m_host1[i].array();\
            m_host2[i].setRandom(rows, cols); \
			m_host2[i] = _Scalar((min) + ((max)-(min))/2) + _Scalar(((max)-(min))/2) * m_host2[i].array();\
		} \
	 \
		cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device1(rows, cols, batches); \
        cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_device2(rows, cols, batches); \
		for (cuMat::Index i = 0; i < batches; ++i) { \
			auto slice1 = cuMat::Matrix<_Scalar, _Rows, _Cols, 1, cuMat::ColumnMajor>::fromEigen(m_host1[i]); \
			m_device1.block(0, 0, i, rows, cols, 1) = slice1; \
            auto slice2 = cuMat::Matrix<_Scalar, _Rows, _Cols, 1, cuMat::ColumnMajor>::fromEigen(m_host2[i]); \
			m_device2.block(0, 0, i, rows, cols, 1) = slice2; \
		} \
	 \
        const auto& X = m_device1; \
        const auto& Y = m_device2; \
		cuMat::Matrix<_Scalar, _Rows, _Cols, _Batches, _Flags> m_deviceRet = cuMatFn; \
	 \
		std::vector<Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value>> m_hostRet(batches); \
		for (cuMat::Index i = 0; i<batches; ++i) \
		{ \
			m_hostRet[i] = m_deviceRet.block(0, 0, i, rows, cols, 1).eval().toEigen(); \
		} \
	 \
		for (cuMat::Index i = 0; i<batches; ++i) \
		{ \
			INFO("Input 1: " << m_host1[i]); \
            INFO("Input 2: " << m_host2[i]); \
            const auto& X = m_host1[i]; \
            const auto& Y = m_host2[i]; \
			Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, cuMat::eigen::StorageCuMatToEigen<_Flags>::value> lhs = eigenFn; \
			auto rhs = m_hostRet[i]; \
			INFO("Expected: " << lhs); \
			INFO("Actual: " << rhs); \
			REQUIRE(lhs.isApprox(rhs, Eigen::NumTraits<_Scalar>::dummy_precision() * rows * cols)); \
		} \
	}

#define BINARY_TEST_CASE_ALL(name, cfn, efn, min, max) \
	BINARY_OP_HELPER(name, cfn, efn, min, max) \
	TEST_CASE(CUMAT_STR(name), "[binary]") \
	{ \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(int, binaryOpHelper_ ## name); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(long, binaryOpHelper_ ## name); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(float, binaryOpHelper_ ## name); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(double, binaryOpHelper_ ## name); \
	}

#define BINARY_TEST_CASE_INT(name, cfn, efn, min, max) \
	BINARY_OP_HELPER(name, cfn, efn, min, max) \
	TEST_CASE(CUMAT_STR(name), "[binary]") \
	{ \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(int, binaryOpHelper_ ## name); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(long, binaryOpHelper_ ## name); \
	}

#define BINARY_TEST_CASE_FLOAT(name, cfn, efn, min, max) \
	BINARY_OP_HELPER(name, cfn, efn, min, max) \
	TEST_CASE(CUMAT_STR(name), "[binary]") \
	{ \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(float, binaryOpHelper_ ## name); \
		CUMAT_TESTS_CALL_SIMPLE_MATRIX_TEST(double, binaryOpHelper_ ## name); \
	}
