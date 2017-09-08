#ifndef __CUMAT_FORWARD_DECLARATIONS_H__
#define __CUMAT_FORWARD_DECLARATIONS_H__

#include "Macros.h"

CUMAT_NAMESPACE_BEGIN

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

CUMAT_NAMESPACE_END

#endif