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
    typedef _Derived Type;
    typedef MatrixBase<_Derived> Base;
    CUMAT_PUBLIC_API_NO_METHODS

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

	typedef Matrix<
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
	eval_t eval() const
	{
		return eval_t(derived());
	}

    /**
     * \brief Conversion: Matrix of size 1-1-1 (scalar) in device memory to the host memory scalar.
     * 
     * This is expecially usefull to directly use the results of full reductions in host code.
     * 
     * \tparam T 
     */
    explicit operator Scalar () const
    {
        CUMAT_STATIC_ASSERT(
            internal::traits<_Derived>::RowsAtCompileTime == 1 &&
            internal::traits<_Derived>::ColsAtCompileTime == 1 &&
            internal::traits<_Derived>::BatchesAtCompileTime == 1,
            "Conversion only possible for compile-time scalars");
        eval_t m = eval();
        Scalar v;
        m.copyToHost(&v);
        return v;
    }


	// CWISE EXPRESSIONS
#include "MatrixBlockPluginRvalue.inl"
#include "UnaryOpsPlugin.inl"
#include "BinaryOpsPlugin.inl"
#include "ReductionOpsPlugin.inl"
#include "DenseLinAlgPlugin.inl"
#include "SparseExpressionOpPlugin.inl"
};



template <typename _Derived, int _AccessFlags>
struct MatrixReadWrapper
{
private:
    enum
    {
        //the existing access flags
        flags = internal::traits<_Derived>::AccessFlags,
        //boolean if the access is sufficient
        sufficient = (flags & _AccessFlags)
    };
public:
    /**
     * \brief The wrapped type: either the type itself, if the access is sufficient,
     * or the evaluated type if not.
     */
    using type = typename std::conditional<bool(sufficient), _Derived, typename _Derived::eval_t>::type;

    /*
    template<typename T = typename std::enable_if<sufficient, MatrixBase<_Derived>>::type>
    static type wrap(const T& m)
    {
        return m.derived();
    }
    template<typename T = typename std::enable_if<!sufficient, MatrixBase<_Derived>>::type>
    static type wrap(const T& m)
    {
        return m.derived().eval();
    }
    */

private:
    MatrixReadWrapper(){} //not constructible
};

CUMAT_NAMESPACE_END

#endif