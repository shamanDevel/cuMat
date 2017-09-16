#ifndef __CUMAT_FORWARD_DECLARATIONS_H__
#define __CUMAT_FORWARD_DECLARATIONS_H__

#include "Macros.h"
#include <cuComplex.h>

CUMAT_NAMESPACE_BEGIN

/**
* \brief complex float type
*/
typedef cuFloatComplex cfloat;
/**
* \brief complex double type
*/
typedef cuDoubleComplex cdouble;

/**
* \brief The datatype used for matrix indexing
*/
typedef ptrdiff_t Index;

//Declares, not defines all types

namespace internal {
	/**
	* \brief Each class that inherits MatrixBase must define a specialization of internal::traits
	* that define the following:
	*
	* \code
	* typedef ... Scalar;
	* enum {
	*	Flags = ...; //storage order, see \ref Flags
	*	RowsAtCompileTime = ...;
	*	ColsAtCompileTime = ...;
	*	BatchesAtCompileTime = ...;
	* };
	* \endcode
	*
	* \tparam T
	*/
	template<typename T> struct traits;

	// here we say once and for all that traits<const T> == traits<T>
	// When constness must affect traits, it has to be constness on template parameters on which T itself depends.
	// For example, traits<Map<const T> > != traits<Map<T> >, but
	//              traits<const Map<T> > == traits<Map<T> >
	// DIRECTLY TAKEN FROM EIGEN
	template<typename T> struct traits<const T> : traits<T> {};
}

template<typename _Derived> class MatrixBase;

template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags> class Matrix;
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _MatrixType> class MatrixBlock;

template<typename _Derived> class CwiseOp;
template<typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _NullaryFunctor> class NullaryOp;
template<typename _Child, typename _UnaryFunctor> class UnaryOp;

namespace functor
{
	template<typename _Scalar> class UnaryMathFunctor_cwiseNegate;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAbs;
	template<typename _Scalar> class UnaryMathFunctor_cwiseInverse;
	template<typename _Scalar> class UnaryMathFunctor_cwiseExp;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLog;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLog1p;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLog10;
	template<typename _Scalar> class UnaryMathFunctor_cwiseSqrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseRsqrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCbrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseRcbrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseSin;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCos;
	template<typename _Scalar> class UnaryMathFunctor_cwiseTan;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAsin;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAcos;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAtan;
	template<typename _Scalar> class UnaryMathFunctor_cwiseSinh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCosh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseTanh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAsinh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAcosh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAtanh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCeil;
	template<typename _Scalar> class UnaryMathFunctor_cwiseFloor;
	template<typename _Scalar> class UnaryMathFunctor_cwiseRound;
	template<typename _Scalar> class UnaryMathFunctor_cwiseErf;
	template<typename _Scalar> class UnaryMathFunctor_cwiseErfc;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLgamma;
}

CUMAT_NAMESPACE_END

#endif