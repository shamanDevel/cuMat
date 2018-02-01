#ifndef __CUMAT_EIGEN_INTEROP_HELPERS_H__
#define __CUMAT_EIGEN_INTEROP_HELPERS_H__


#include "Macros.h"
#include "Constants.h"

#if CUMAT_EIGEN_SUPPORT==1
#include <complex>
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

	//Type conversion
	template<typename T>
	struct TypeCuMatToEigen
	{
		typedef T type;
	};
	template<>
	struct TypeCuMatToEigen<cfloat>
	{
		typedef std::complex<float> type;
	};
	template<>
	struct TypeCuMatToEigen<cdouble>
	{
		typedef std::complex<double> type;
	};

	template<typename T>
	struct TypeEigenToCuMat
	{
		typedef T type;
	};
	template<>
	struct TypeEigenToCuMat<std::complex<float> >
	{
		typedef cfloat type;
	};
	template<>
	struct TypeEigenToCuMat<std::complex<double> >
	{
		typedef cdouble type;
	};

	//Matrix type conversion

	template<typename _CuMatMatrixType>
	struct MatrixCuMatToEigen
	{
		using type = ::Eigen::Matrix<
			//typename TypeCuMatToEigen<typename _CuMatMatrixType::Scalar>::type,
			typename _CuMatMatrixType::Scalar,
			SizeCuMatToEigen<_CuMatMatrixType::Rows>::size,
			SizeCuMatToEigen<_CuMatMatrixType::Columns>::size,
            //Eigen requires specific storage types for vector sizes
			((_CuMatMatrixType::Rows==1) ? ::Eigen::StorageOptions::RowMajor  
            : (_CuMatMatrixType::Columns==1) ? ::Eigen::StorageOptions::ColMajor
	        : StorageCuMatToEigen<_CuMatMatrixType::Flags>::value)
	        | ::Eigen::DontAlign //otherwise, toEigen() will produce strange errors because we access the native data pointer
		>;
	};
	template<typename _EigenMatrixType>
	struct MatrixEigenToCuMat
	{
		using type = ::cuMat::Matrix<
			//typename TypeEigenToCuMat<typename _EigenMatrixType::Scalar>::type,
			typename _EigenMatrixType::Scalar,
			SizeEigenToCuMat<_EigenMatrixType::RowsAtCompileTime>::size,
			SizeEigenToCuMat<_EigenMatrixType::ColsAtCompileTime>::size,
			1, //batch size of 1
			StorageEigenToCuMat<_EigenMatrixType::Options>::value
		>;
	};

}

CUMAT_NAMESPACE_END

//tell Eigen how to handle cfloat and cdouble
namespace Eigen
{
    template<> struct NumTraits<CUMAT_NAMESPACE cfloat> : NumTraits<std::complex<float>> {};
    template<> struct NumTraits<CUMAT_NAMESPACE cdouble> : NumTraits<std::complex<double>> {};
}


#endif

#endif
