#ifndef __CUMAT_EIGEN_INTEROP_HELPERS_H__
#define __CUMAT_EIGEN_INTEROP_HELPERS_H__


#include "Macros.h"
#include "Constants.h"

#if CUMAT_EIGEN_SUPPORT==1
#include <Eigen/Core>

CUMAT_NAMESPACE_BEGIN

//forward-declare Matrix
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
class Matrix;

/**
 * Namespace for eigen interop.
 */
namespace eigen
{
	//There are now a lot of redundant qualifiers, but I want to make
	//it clear if we are in the Eigen world or in cuMat world.

	//Flag conversion

	template<int _Flags>
	struct StorageCuMatToEigen {};
	template<>
	struct StorageCuMatToEigen<::cuMat::Flags::ColumnMajor>
	{
		enum { value = ::Eigen::StorageOptions::ColMajor };
	};
	template<>
	struct StorageCuMatToEigen<::cuMat::Flags::RowMajor>
	{
		enum { value = ::Eigen::StorageOptions::RowMajor };
	};

	template<int _Storage>
	struct StorageEigenToCuMat {};
	template<>
	struct StorageEigenToCuMat<::Eigen::StorageOptions::RowMajor>
	{
		enum { value = ::cuMat::Flags::RowMajor };
	};
	template<>
	struct StorageEigenToCuMat<::Eigen::StorageOptions::ColMajor>
	{
		enum { value = ::cuMat::Flags::ColumnMajor };
	};

	//Size conversion (for dynamic tag)

	template<int _Size>
	struct SizeCuMatToEigen
	{
		enum { size = _Size };
	};
	template<>
	struct SizeCuMatToEigen<::cuMat::Dynamic>
	{
		enum {size = ::Eigen::Dynamic};
	};

	template<int _Size>
	struct SizeEigenToCuMat
	{
		enum {size = _Size};
	};
	template<>
	struct SizeEigenToCuMat<::Eigen::Dynamic>
	{
		enum{size = ::cuMat::Dynamic};
	};

	//Matrix type conversion

	template<typename _CuMatMatrixType>
	struct MatrixCuMatToEigen
	{
		using type = ::Eigen::Matrix<
			typename _CuMatMatrixType::Scalar,
			SizeCuMatToEigen<_CuMatMatrixType::Rows>::size,
			SizeCuMatToEigen<_CuMatMatrixType::Columns>::size,
			StorageCuMatToEigen<_CuMatMatrixType::Flags>::value
		>;
	};
	template<typename _EigenMatrixType>
	struct MatrixEigenToCuMat
	{
		using type = ::cuMat::Matrix<
			typename _EigenMatrixType::Scalar,
			SizeEigenToCuMat<_EigenMatrixType::RowsAtCompileTime>::size,
			SizeEigenToCuMat<_EigenMatrixType::ColsAtCompileTime>::size,
			1, //batch size of 1
			StorageEigenToCuMat<_EigenMatrixType::Options>::value
		>;
	};
}

CUMAT_NAMESPACE_END

#endif

#endif