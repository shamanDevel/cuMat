#ifndef __CUMAT_MATRIX_BASE_H__
#define __CUMAT_MATRIX_BASE_H__

#include <cuda_runtime.h>

#include "Macros.h"
#include "Constants.h"

CUMAT_NAMESPACE_BEGIN

template<typename Derived>
class MatrixBase
{
	
public:
	/** 
	 * \returns a reference to the derived object 
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Derived& derived() { return *static_cast<Derived*>(this); }
	
	/** 
	 * \returns a const reference to the derived object 
	 */
	__host__ __device__ CUMAT_STRONG_INLINE const Derived& derived() const { return *static_cast<const Derived*>(this); }

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

	// TODO: eval()
};

CUMAT_NAMESPACE_END

#endif