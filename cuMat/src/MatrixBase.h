#ifndef __CUMAT_MATRIX_BASE_H__
#define __CUMAT_MATRIX_BASE_H__

#include <cuda_runtime.h>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Constants.h"

CUMAT_NAMESPACE_BEGIN

/**
 * \brief The base class of all matrix types and matrix expressions.
 * \tparam _Derived 
 */
template<typename _Derived>
class MatrixBase
{
public:

	typedef typename internal::traits<_Derived>::Scalar Scalar;

	/** 
	 * \returns a reference to the _Derived object 
	 */
	__host__ __device__ CUMAT_STRONG_INLINE _Derived& derived() { return *static_cast<_Derived*>(this); }
	
	/** 
	 * \returns a const reference to the _Derived object 
	 */
	__host__ __device__ CUMAT_STRONG_INLINE const _Derived& derived() const { return *static_cast<const _Derived*>(this); }

	/** 
	 * \brief Returns the number of rows of this matrix.
	 * \returns the number of rows.
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return derived().rows(); }
	/**
	 * \brief Returns the number of columns of this matrix.
	 * \returns the number of columns.
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return derived().cols(); }
	/**
	 * \brief Returns the number of batches of this matrix.
	 * \returns the number of batches.
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return derived().batches(); }
	/**
	* \brief Returns the total number of entries in this matrix.
	* This value is computed as \code rows()*cols()*batches()* \endcode
	* \return the total number of entries
	*/
	__host__ __device__ CUMAT_STRONG_INLINE Index size() const { return rows()*cols()*batches(); }

	// EVALUATION

	typedef typename Matrix<
		typename internal::traits<_Derived>::Scalar,
		internal::traits<_Derived>::RowsAtCompileTime,
		internal::traits<_Derived>::ColsAtCompileTime,
		internal::traits<_Derived>::BatchesAtCompileTime,
		internal::traits<_Derived>::Flags
	> eval_t;

	/**
	 * \brief Evaluates this into a matrix.
	 * This evaluates any expression template. If this is already a matrix, it is returned unchanged.
	 * \return the evaluated matrix
	 */
	eval_t eval()
	{
		return eval_t(derived());
	}

	/**
	 * \brief Evaluates this into the the specified matrix.
	 * This is called within eval() and the constructor of Matrix,
	 * don't call it in user-code.
	 */
	template<typename Derived>
	void evalTo(MatrixBase<Derived>& m) const { derived().evalTo(m); }


	// CWISE EXPRESSIONS
#include "UnaryOpsPlugin.h"
#include "BinaryOpsPlugin.h"
};

CUMAT_NAMESPACE_END

#endif