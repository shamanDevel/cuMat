#ifndef __CUMAT_TESTS_UTILS_H__
#define __CUMAT_TESTS_UTILS_H__

#include <catch/catch.hpp>
#include <cuMat/src/Matrix.h>


#define __CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, _scalar, _rows, _cols, _batches, _flags, rows, cols, batches) \
	{ \
		INFO("Run " << #Test << ": Scalar=" << #_scalar << ", RowsAtCompileTime=" << #_rows << ", ColsAtCompileTime=" << #_cols << ", BatchesAtCompileTime=" << #_batches << ", Flags=" << #_flags << ", Rows=" << rows << ", Cols=" << cols << ", Batches=" << batches); \
		Test<_scalar, _rows, _cols, _batches, _flags>(rows, cols, batches); \
	}

/**
 * \brief Tests a function template 'Test' based on different settings for sizes.
 * The function 'Test' must be defined as follows:
 * \code
 * template<typename _Scalar, int _Rows, int _Cols, int _Batches, int _Flags>
 * void MyTestFunction(Index rows, Index cols, Index batches) {...};
 * \endcode
 * And then you can call it on floating point values e.g. by
 * \code
 * CUMAT_TESTS_CALL_MATRIX_TEST(float, MyTestFunction);
 * CUMAT_TESTS_CALL_MATRIX_TEST(double, MyTestFunction);
 * \endcode
 * \param Scalar the scalar type
 * \param Test the test function
 */
#define CUMAT_TESTS_CALL_MATRIX_TEST(Scalar, Test) \
{ \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, 12, 16, cuMat::RowMajor, 8, 12, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, 12, 16, cuMat::RowMajor, 20, 12, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, cuMat::Dynamic, 16, cuMat::RowMajor, 8, 20, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, 12, cuMat::Dynamic, cuMat::RowMajor, 8, 12, 20); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, cuMat::Dynamic, 16, cuMat::RowMajor, 20, 25, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, 12, cuMat::Dynamic, cuMat::RowMajor, 25, 12, 20); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor, 8, 20, 25); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor, 10, 8, 12); \
	\
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, 12, 16, cuMat::ColumnMajor, 8, 12, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, 12, 16, cuMat::ColumnMajor, 20, 12, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, cuMat::Dynamic, 16, cuMat::ColumnMajor, 8, 20, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, 12, cuMat::Dynamic, cuMat::ColumnMajor, 8, 12, 20); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, cuMat::Dynamic, 16, cuMat::ColumnMajor, 20, 25, 16); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, 12, cuMat::Dynamic, cuMat::ColumnMajor, 25, 12, 20); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, 8, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor, 8, 20, 25); \
	__CUMAT_TESTS_CALL_SINGLE_MATRIX_TEST(Test, Scalar, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor, 10, 8, 12); \
}

#endif