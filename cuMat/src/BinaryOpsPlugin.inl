//Included inside MatrixBase, define the accessors

#define BINARY_OP_ACCESSOR(Name) \
    template<typename _Right> \
	BinaryOp<_Derived, _Right, functor::BinaryMathFunctor_ ## Name <Scalar> > Name (const MatrixBase<_Right>& rhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<_Derived, _Right, functor::BinaryMathFunctor_ ## Name <Scalar> >(derived(), rhs.derived()); \
	} \
    template<typename _Right, typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<_Right, Scalar>::value, \
        BinaryOp<_Derived, HostScalar<Scalar>, functor::BinaryMathFunctor_ ## Name <Scalar> > >::type > \
    T Name(const _Right& rhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<_Derived, HostScalar<Scalar>, functor::BinaryMathFunctor_ ## Name <Scalar> >(derived(), HostScalar<Scalar>(rhs)); \
	}
#define BINARY_OP_ACCESSOR_INV(Name) \
    template<typename _Left> \
        BinaryOp<_Left, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> > Name ## Inv(const MatrixBase<_Left>& lhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<_Left, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> >(lhs.derived(), derived()); \
	} \
    template<typename _Left, typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<_Left, Scalar>::value, \
        BinaryOp<HostScalar<Scalar>, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> > >::type > \
    T Name ## Inv(const _Left& lhs) const { \
		CUMAT_ERROR_IF_NO_NVCC(Name)  \
		return BinaryOp<HostScalar<Scalar>, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar> >(HostScalar<Scalar>(lhs), derived()); \
	}

/**
* \brief computes the component-wise multiplation (this*rhs)
*/
BINARY_OP_ACCESSOR(cwiseMul)

/**
* \brief computes the component-wise dot-product. 
* The dot product of every individual element is computed,
*  for regular matrices/vectors with scalar entries,
*  this is exactly equal to the component-wise multiplication (\ref cwiseMul).
* However, one can also use Matrix with submatrices/vectors as entries
*  and then this operation might have the real dot-product
*  if the respective functor \ref functor::BinaryMathFunctor_cwiseMDot is specialized.
*/
BINARY_OP_ACCESSOR(cwiseDot)

/**
* \brief computes the component-wise division (this/rhs)
*/
BINARY_OP_ACCESSOR(cwiseDiv)

/**
* \brief computes the inverted component-wise division (rhs/this)
*/
BINARY_OP_ACCESSOR_INV(cwiseDiv)

/**
* \brief computes the component-wise exponent (this^rhs)
*/
BINARY_OP_ACCESSOR(cwisePow)

/**
* \brief computes the inverted component-wise exponent (rhs^this)
*/
BINARY_OP_ACCESSOR_INV(cwisePow)

/**
* \brief computes the component-wise binary AND (this & rhs).
* Only available for integer matrices.
*/
BINARY_OP_ACCESSOR(cwiseBinaryAnd)

/**
* \brief computes the component-wise binary OR (this | rhs).
* Only available for integer matrices.
*/
BINARY_OP_ACCESSOR(cwiseBinaryOr)

/**
* \brief computes the component-wise binary XOR (this ^ rhs).
* Only available for integer matrices.
*/
BINARY_OP_ACCESSOR(cwiseBinaryXor)

/**
* \brief computes the component-wise logical AND (this && rhs).
* Only available for boolean matrices
*/
BINARY_OP_ACCESSOR(cwiseLogicalAnd)

/**
* \brief computes the component-wise logical OR (this || rhs).
* Only available for boolean matrices
*/
BINARY_OP_ACCESSOR(cwiseLogicalOr)

/**
* \brief computes the component-wise logical XOR (this ^ rhs).
* Only available for boolean matrices
*/
BINARY_OP_ACCESSOR(cwiseLogicalXor)

#undef BINARY_OP_ACCESSOR
#undef BINARY_OP_ACCESSOR_INV

/**
 * \brief Custom binary expression.
 * The binary functor must support look as follow:
 * \code
 * struct MyFunctor
 * {
 *     typedef OutputType ReturnType;
 *     __device__ CUMAT_STRONG_INLINE ReturnType operator()(const LeftType& x, const RightType& y, Index row, Index col, Index batch) const
 *     {
 *         return ...
 *     }
 * };
 * \endcode
 * where \c LeftType is the type of this matrix expression,
 * \c RightType is the type of the rhs matrix,
 * and \c OutputType is the output type.
 *
 * \param rhs the matrix expression on the right hand side
 * \param functor the functor to apply component-wise
 * \return an expression of a component-wise binary expression with a custom functor applied per component.
 */
template<typename Right, typename Functor>
UnaryOp<_Derived, Functor> binaryExpr(const Right& rhs, const Functor& functor = Functor()) const
{
	CUMAT_ERROR_IF_NO_NVCC(binaryExpr)
    return BinaryOp<_Derived, Right, Functor>(derived(), rhs.derived(), functor);
}