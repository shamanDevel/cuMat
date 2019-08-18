//Included inside MatrixBase, define the accessors

#define UNARY_OP_ACCESSOR(Name) \
	UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> > Name () const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> >(derived()); \
	}

/**
 * \brief computes the component-wise negation (x -> -x)
 */
UNARY_OP_ACCESSOR(cwiseNegate);
/**
* \brief computes the component-wise absolute value (x -> |x|)
*/
UNARY_OP_ACCESSOR(cwiseAbs);
/**
* \brief squares the value (x -> x^2)
*/
UNARY_OP_ACCESSOR(cwiseAbs2);
/**
* \brief computes the component-wise inverse (x -> 1/x)
*/
UNARY_OP_ACCESSOR(cwiseInverse);
/**
* \brief computes the component-wise inverse (x -> 1/x)
* with the additional check that it returns 1 if x is zero.
*/
UNARY_OP_ACCESSOR(cwiseInverseCheck);

/**
* \brief computes the component-wise exponent (x -> exp(x))
*/
UNARY_OP_ACCESSOR(cwiseExp);
/**
* \brief computes the component-wise natural logarithm (x -> log(x))
*/
UNARY_OP_ACCESSOR(cwiseLog);
/**
* \brief computes the component-wise value of (x -> log(x+1))
*/
UNARY_OP_ACCESSOR(cwiseLog1p);
/**
* \brief computes the component-wise value of (x -> log_10(x))
*/
UNARY_OP_ACCESSOR(cwiseLog10);

/**
* \brief computes the component-wise square root (x -> sqrt(x))
*/
UNARY_OP_ACCESSOR(cwiseSqrt);
/**
* \brief computes the component-wise reciprocal square root (x -> 1 / sqrt(x))
*/
UNARY_OP_ACCESSOR(cwiseRsqrt);
/**
* \brief computes the component-wise cube root (x -> x^(1/3))
*/
UNARY_OP_ACCESSOR(cwiseCbrt);
/**
* \brief computes the component-wise reciprocal cube root (x -> x^(-1/3))
*/
UNARY_OP_ACCESSOR(cwiseRcbrt);

/**
* \brief computes the component-wise value of (x -> sin(x))
*/
UNARY_OP_ACCESSOR(cwiseSin);
/**
* \brief computes the component-wise value of (x -> cos(x))
*/
UNARY_OP_ACCESSOR(cwiseCos);
/**
* \brief computes the component-wise value of (x -> tan(x))
*/
UNARY_OP_ACCESSOR(cwiseTan);
/**
* \brief computes the component-wise value of (x -> asin(x))
*/
UNARY_OP_ACCESSOR(cwiseAsin);
/**
* \brief computes the component-wise value of (x -> acos(x))
*/
UNARY_OP_ACCESSOR(cwiseAcos);
/**
* \brief computes the component-wise value of (x -> atan(x))
*/
UNARY_OP_ACCESSOR(cwiseAtan);
/**
* \brief computes the component-wise value of (x -> sinh(x))
*/
UNARY_OP_ACCESSOR(cwiseSinh);
/**
* \brief computes the component-wise value of (x -> cosh(x))
*/
UNARY_OP_ACCESSOR(cwiseCosh);
/**
* \brief computes the component-wise value of (x -> tanh(x))
*/
UNARY_OP_ACCESSOR(cwiseTanh);
/**
* \brief computes the component-wise value of (x -> asinh(x))
*/
UNARY_OP_ACCESSOR(cwiseAsinh);
/**
* \brief computes the component-wise value of (x -> acosh(x))
*/
UNARY_OP_ACCESSOR(cwiseAcosh);
/**
* \brief computes the component-wise value of (x -> atanh(x))
*/
UNARY_OP_ACCESSOR(cwiseAtanh);
/**
* \brief Component-wise rounds up the entries to the next larger integer.
* For an integer matrix, this does nothing
*/
UNARY_OP_ACCESSOR(cwiseCeil);
/**
* \brief Component-wise rounds down the entries to the next smaller integer.
* For an integer matrix, this does nothing
*/
UNARY_OP_ACCESSOR(cwiseFloor);
/**
* \brief Component-wise rounds the entries to the next integer.
* For an integer matrix, this does nothing
*/
UNARY_OP_ACCESSOR(cwiseRound);
/**
* \brief Calculate the error function of the input argument component-wise (x -> erf(x))
*/
UNARY_OP_ACCESSOR(cwiseErf);
/**
* \brief Calculate the complementary error function of the input argument component-wise (x -> erfc(x))
*/
UNARY_OP_ACCESSOR(cwiseErfc);
/**
* \brief Calculate the natural logarithm of the absolute value of the gamma function of the input argument component-wise (x -> lgamma(x))
*/
UNARY_OP_ACCESSOR(cwiseLgamma);

/**
* \brief Calculate the component-wise binary negation (x -> ~x).
* Only available for integer matrices.
*/
UNARY_OP_ACCESSOR(cwiseBinaryNot);

/**
* \brief Calculate the component-wise logical negation (x -> !x).
* Only available for boolean matrices.
*/
UNARY_OP_ACCESSOR(cwiseLogicalNot);

/**
* \brief Conjugates the matrix. This is a no-op for non-complex matrices.
*/
UNARY_OP_ACCESSOR(conjugate);

#undef UNARY_OP_ACCESSOR

/**
 * \brief Negates this matrix
 */
UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate<Scalar> > operator-() const {
	CUMAT_ERROR_IF_NO_NVCC(negate)
	return UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate <Scalar> >(derived());
}

/**
 * \brief Custom unary expression.
 * The unary functor must look as follow:
 * \code
 * struct MyFunctor
 * {
 *     typedef OutputType ReturnType;
 *     __device__ CUMAT_STRONG_INLINE ReturnType operator()(const InputType& v, Index row, Index col, Index batch) const
 *     {
 *         return ...
 *     }
 * };
 * \endcode
 * with \c InputType being the type of this matrix expression and \c OutputType the output type.
 */
template<typename Functor>
UnaryOp<_Derived, Functor> unaryExpr(const Functor& functor = Functor()) const
{
	CUMAT_ERROR_IF_NO_NVCC(unaryExpr)
    return UnaryOp<_Derived, Functor>(derived(), functor);
}

/**
 * \brief Transposes this matrix
 */
TransposeOp<_Derived, false> transpose() const
{
	return TransposeOp<_Derived, false>(derived());
}

/**
* \brief Returns the adjoint of this matrix (the conjugated transpose).
*/
TransposeOp<_Derived, true> adjoint() const
{
    return TransposeOp<_Derived, true>(derived());
}

/**
 * \brief Casts this matrix into a matrix of the target datatype
 * \tparam _Target the target type
 */
template<typename _Target>
CastingOp<_Derived, _Target> cast() const
{
	CUMAT_ERROR_IF_NO_NVCC(cast)
	return CastingOp<_Derived, _Target>(derived());
}

/**
 * \brief Returns a diagonal matrix with this vector as the main diagonal.
 * This is only available for compile-time row- or column vectors.
 */
AsDiagonalOp<_Derived> asDiagonal() const
{
	CUMAT_ERROR_IF_NO_NVCC(asDiagonal)
    return AsDiagonalOp<_Derived>(derived());
}

/**
 * \brief Extracts the main diagonal of this matrix and returns it as a column vector.
 * The matrix must not necessarily be square.
 */
ExtractDiagonalOp<_Derived> diagonal() const
{
	CUMAT_ERROR_IF_NO_NVCC(diagonal)
    return ExtractDiagonalOp<_Derived>(derived());
}

#ifdef CUMAT_PARSED_BY_DOXYGEN

/**
 * \brief Extracts the real part of the complex matrix.
 * On real matrices, this is a no-op.
 */
ExtractComplexPartOp<_Derived, false, false> real() const
{
    return ExtractComplexPartOp<_Derived, false, false>(derived());
}

#else

/**
 * \brief Extracts the real part of the complex matrix.
 * Specialization for real matrices: no-op.
 */
template<typename S = typename internal::traits<_Derived>::Scalar,
         typename = typename std::enable_if<!internal::NumTraits<S>::IsComplex>::type>
_Derived real() const
{
    return derived();
}

/**
 * \brief Extracts the real part of the complex matrix.
 * Specialization for complex matrices.
 */
template<typename S = typename internal::traits<_Derived>::Scalar,
         typename = typename std::enable_if<internal::NumTraits<S>::IsComplex>::type>
ExtractComplexPartOp<_Derived, false, false> real() const
{
	CUMAT_ERROR_IF_NO_NVCC(real)
    CUMAT_STATIC_ASSERT(internal::NumTraits<typename internal::traits<_Derived>::Scalar>::IsComplex, "Matrix must be complex");
    return ExtractComplexPartOp<_Derived, false, false>(derived());
}

#endif

/**
 * \brief Extracts the imaginary part of the complex matrix.
 * This method is only available for complex matrices.
 */
ExtractComplexPartOp<_Derived, true, false> imag() const
{
	CUMAT_ERROR_IF_NO_NVCC(imag)
    CUMAT_STATIC_ASSERT(internal::NumTraits<typename internal::traits<_Derived>::Scalar>::IsComplex, "Matrix must be complex");
    return ExtractComplexPartOp<_Derived, true, false>(derived());
}

/**
 * \brief Swaps the axis of this matrix.
 * 
 * This operation is best explained on examples:
 *  - <code>matrix.swapAxis<Column, Row, Batch>()</code> 
 *    returns the component-wise transpose
 *  - <code>batchedVector.swapAxis<Row, Batch, NoAxis>()</code>
 *    pulls in the batch dimension into the columns of the matrix.
 *    The batch dimension is removed.
 *  - <code>vector.swapAxis<NoAxis, NoAxis, Row>()</code>
 *    converts the vector to a batched scalar tensor.
 * 
 * \tparam _Row the axis which is used as the new row index
 * \tparam _Col the axis which is used as the new column index
 * \tparam _Batch the axis which is used as the new batch index
 */
template<Axis _Row, Axis _Col, Axis _Batch>
SwapAxisOp<_Derived, _Row, _Col, _Batch> swapAxis() const
{
	CUMAT_ERROR_IF_NO_NVCC(swapAxis)
	return SwapAxisOp<_Derived, _Row, _Col, _Batch>(derived());
}
