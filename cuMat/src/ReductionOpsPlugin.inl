//Included inside MatrixBase, define the accessors

/**
 * \brief Computes the sum of all elements along the specified reduction axis
 * \tparam axis the reduction axis, by default, reduction is performed among all axis
 * \tparam Algorithm the reduction algorithm, a tag from the namespace ReductionAlg
 */
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::Sum<Scalar>, axis, Algorithm> sum() const
{
	CUMAT_ERROR_IF_NO_NVCC(sum)
    return ReductionOp_StaticSwitched<_Derived, functor::Sum<Scalar>, axis, Algorithm>(
		derived(), functor::Sum<Scalar>(), 0);
}

/**
* \brief Computes the sum of all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::Sum<Scalar>, Algorithm> sum(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(sum)
    return ReductionOp_DynamicSwitched<_Derived, functor::Sum<Scalar>, Algorithm>(
		derived(), axis, functor::Sum<Scalar>(), 0);
}

/**
* \brief Computes the product of all elements along the specified reduction axis
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::Prod<Scalar>, axis, Algorithm> prod() const
{
	CUMAT_ERROR_IF_NO_NVCC(prod)
    return ReductionOp_StaticSwitched<_Derived, functor::Prod<Scalar>, axis, Algorithm>(
		derived(), functor::Prod<Scalar>(), 1);
}

/**
* \brief Computes the product of all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::Prod<Scalar>, Algorithm> prod(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(prod)
    return ReductionOp_DynamicSwitched<_Derived, functor::Prod<Scalar>, Algorithm>(
		derived(), axis, functor::Prod<Scalar>(), 1);
}

/**
* \brief Computes the minimum value among all elements along the specified reduction axis
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::Min<Scalar>, axis, Algorithm> minCoeff() const
{
	CUMAT_ERROR_IF_NO_NVCC(minCoeff)
    return ReductionOp_StaticSwitched<_Derived, functor::Min<Scalar>, axis, Algorithm>(
		derived(), functor::Min<Scalar>(), std::numeric_limits<Scalar>::max());
}

/**
* \brief Computes the minimum value among all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::Min<Scalar>, Algorithm> minCoeff(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(minCoeff)
    return ReductionOp_DynamicSwitched<_Derived, functor::Min<Scalar>, Algorithm>(
		derived(), axis, functor::Min<Scalar>(), std::numeric_limits<Scalar>::max());
}

/**
* \brief Computes the maximum value among all elements along the specified reduction axis
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::Max<Scalar>, axis, Algorithm> maxCoeff() const
{
	CUMAT_ERROR_IF_NO_NVCC(maxCoeff)
    return ReductionOp_StaticSwitched<_Derived, functor::Max<Scalar>, axis, Algorithm>(
		derived(), functor::Max<Scalar>(), std::numeric_limits<Scalar>::lowest());
}

/**
* \brief Computes the maximum value among all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::Max<Scalar>, Algorithm> maxCoeff(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(maxCoeff)
    return ReductionOp_DynamicSwitched<_Derived, functor::Max<Scalar>, Algorithm>(
		derived(), axis, functor::Max<Scalar>(), std::numeric_limits<Scalar>::lowest());
}

/**
* \brief Computes the locical AND of all elements along the specified reduction axis,
* i.e. <b>all</b> values must be true for the result to be true.
* This is only defined for boolean matrices.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::LogicalAnd<Scalar>, axis, Algorithm> all() const
{
	CUMAT_ERROR_IF_NO_NVCC(all)
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'all' can only be applied to boolean matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::LogicalAnd<Scalar>, axis, Algorithm>(
		derived(), functor::LogicalAnd<Scalar>(), true);
}

/**
* \brief Computes the logical AND of all elements along the specified reduction axis,
* i.e. <b>all</b> values must be true for the result to be true.
* This is only defined for boolean matrices.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::LogicalAnd<Scalar>, Algorithm> all(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(all)
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'all' can only be applied to boolean matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::LogicalAnd<Scalar>, Algorithm>(
		derived(), axis, functor::LogicalAnd<Scalar>(), true);
}

/**
* \brief Computes the locical OR of all elements along the specified reduction axis,
* i.e. <b>any</b> value must be true for the result to be true.
* This is only defined for boolean matrices.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::LogicalOr<Scalar>, axis, Algorithm> any() const
{
	CUMAT_ERROR_IF_NO_NVCC(any)
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'any' can only be applied to boolean matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::LogicalOr<Scalar>, axis, Algorithm>(
		derived(), functor::LogicalOr<Scalar>(), false);
}

/**
* \brief Computes the logical OR of all elements along the specified reduction axis,
* i.e. <b>any</b> values must be true for the result to be true.
* This is only defined for boolean matrices.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::LogicalOr<Scalar>, Algorithm> any(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(any)
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'any' can only be applied to boolean matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::LogicalOr<Scalar>, Algorithm>(
		derived(), axis, functor::LogicalOr<Scalar>(), false);
}

/**
* \brief Computes the bitwise AND of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::BitwiseAnd<Scalar>, axis, Algorithm> bitwiseAnd() const
{
	CUMAT_ERROR_IF_NO_NVCC(bitwiseAnd)
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseAnd' can only be applied to integral matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::BitwiseAnd<Scalar>, axis, Algorithm>(
		derived(), functor::BitwiseAnd<Scalar>(), ~(Scalar(0)));
}

/**
* \brief Computes the logical AND of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::BitwiseAnd<Scalar>, Algorithm> bitwiseAnd(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(bitwiseAnd)
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseAnd' can only be applied to integral matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::BitwiseAnd<Scalar>, Algorithm>(
		derived(), axis, functor::BitwiseAnd<Scalar>(), ~(Scalar(0)));
}

/**
* \brief Computes the bitwise OR of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = Axis::Row | Axis::Column | Axis::Batch, typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, functor::BitwiseOr<Scalar>, axis, Algorithm> bitwiseOr() const
{
	CUMAT_ERROR_IF_NO_NVCC(bitwiseOr)
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseOr' can only be applied to integral matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::BitwiseOr<Scalar>, axis, Algorithm>(
		derived(), functor::BitwiseOr<Scalar>(), Scalar(0));
}

/**
* \brief Computes the logical OR of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, functor::BitwiseOr<Scalar>, Algorithm> bitwiseOr(int axis) const
{
	CUMAT_ERROR_IF_NO_NVCC(bitwiseOr)
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseOr' can only be applied to integral matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::BitwiseOr<Scalar>, Algorithm>(
		derived(), axis, functor::BitwiseOr<Scalar>(), Scalar(0));
}

//custom reduction
/**
* \brief Custom reduction operation (static axis).
* Here you can pass your own reduction operator and initial value.
*   
* \param functor the reduction functor
* \param initialValue the initial value to the reduction 
* 
* \tparam _Functor the reduction functor, must suppor the operation
*   \code __device__ T operator()(const T &a, const T &b) const \endcode
*   with \c T being the current scalar type
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<
    typename _Functor,
    int axis = Axis::Row | Axis::Column | Axis::Batch,
	typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<_Derived, _Functor, axis, Algorithm>
reduction(const _Functor& functor = _Functor(), const Scalar& initialValue = Scalar(0)) const
{
	CUMAT_ERROR_IF_NO_NVCC(reduction)
    return ReductionOp_StaticSwitched<_Derived, _Functor, axis, Algorithm>(
		derived(), functor, initialValue);
}

/**
* \brief Custom reduction operation (dynamic axis).
* Here you can pass your own reduction operator and initial value.
* 
* \param axis the reduction axis, a combination of the constants in \ref Axis
* \param functor the reduction functor
* \param initialValue the initial value to the reduction 
* 
* \tparam _Functor the reduction functor, must support the operation
*   \code __device__ T operator()(const T &a, const T &b) const \endcode
*   with \c T being the current scalar type
*/
template<typename _Functor, typename Algorithm = ReductionAlg::Auto>
ReductionOp_DynamicSwitched<_Derived, _Functor, Algorithm>
reduction(int axis, const _Functor& functor = _Functor(), const Scalar& initialValue = Scalar(0)) const
{
	CUMAT_ERROR_IF_NO_NVCC(reduction)
    return ReductionOp_DynamicSwitched<_Derived, _Functor, Algorithm>(
		derived(), axis, functor, initialValue);
}

//combined ops

/**
 * \brief Computes the trace of the matrix.
 * This is simply implemented as <tt>*this.diagonal().sum<Axis::Column>()</tt>
 */
template<typename Algorithm = ReductionAlg::Auto>
ReductionOp_StaticSwitched<
	ExtractDiagonalOp<_Derived>, 
	functor::Sum<Scalar>, 
	Axis::Row | Axis::Column,
	Algorithm> trace() const
{
	CUMAT_ERROR_IF_NO_NVCC(trace)
    return diagonal().template sum<Axis::Row | Axis::Column, Algorithm>();
}

template<typename _Other, typename Algorithm = ReductionAlg::Auto>
using DotReturnType = ReductionOp_StaticSwitched<
	BinaryOp<_Derived, _Other, functor::BinaryMathFunctor_cwiseDot<Scalar> >, 
	functor::Sum<typename functor::BinaryMathFunctor_cwiseDot<Scalar>::ReturnType>, 
	Axis::Row | Axis::Column,
	Algorithm>;
/**
 * \brief Computes the dot product between two vectors.
 * This method is only allowed on compile-time vectors of the same orientation (either row- or column vector).
 */
template<typename _Other, typename Algorithm = ReductionAlg::Auto>
DotReturnType<_Other, Algorithm> dot(const MatrixBase<_Other>& rhs) const
{
	CUMAT_ERROR_IF_NO_NVCC(dot)
    CUMAT_STATIC_ASSERT(internal::traits<_Derived>::RowsAtCompileTime == 1 || internal::traits<_Derived>::ColsAtCompileTime == 1,
        "This matrix must be a compile-time row or column vector");
    CUMAT_STATIC_ASSERT(internal::traits<_Other>::RowsAtCompileTime == 1 || internal::traits<_Other>::ColsAtCompileTime == 1,
        "The right-hand-side must be a compile-time row or column vector");
    return ((*this).cwiseDot(rhs)).template sum<Axis::Row | Axis::Column, Algorithm>();
}

template<typename Algorithm = ReductionAlg::Auto>
using SquaredNormReturnType = ReductionOp_StaticSwitched<
	UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseAbs2<Scalar> >, 
	functor::Sum<typename functor::UnaryMathFunctor_cwiseAbs2<Scalar>::ReturnType>, 
	Axis::Row | Axis::Column,
	Algorithm>;
/**
 * \brief Computes the squared l2-norm of this matrix if it is a vecotr, or the squared Frobenius norm if it is a matrix.
 * It consists in the the sum of the square of all the matrix entries.
 */
template<typename Algorithm = ReductionAlg::Auto>
SquaredNormReturnType<Algorithm> squaredNorm() const
{
	CUMAT_ERROR_IF_NO_NVCC(squaredNorm)
    return cwiseAbs2().template sum<Axis::Row | Axis::Column, Algorithm>();
}

template<typename Algorithm = ReductionAlg::Auto>
using NormReturnType = UnaryOp<
	ReductionOp_StaticSwitched<
		UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseAbs2<Scalar> >, 
		functor::Sum<typename functor::UnaryMathFunctor_cwiseAbs2<Scalar>::ReturnType>, 
		Axis::Row | Axis::Column,
		Algorithm>,
	functor::UnaryMathFunctor_cwiseSqrt<typename functor::UnaryMathFunctor_cwiseAbs2<Scalar>::ReturnType> >;
/**
 * \brief Computes the l2-norm of this matrix if it is a vecotr, or the Frobenius norm if it is a matrix.
 * It consists in the square root of the sum of the square of all the matrix entries.
 */
template<typename Algorithm = ReductionAlg::Auto>
NormReturnType<Algorithm> norm() const
{
	CUMAT_ERROR_IF_NO_NVCC(norm)
    return squaredNorm<Algorithm>().cwiseSqrt();
}

