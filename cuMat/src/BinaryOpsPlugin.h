//Included inside MatrixBase, define the accessors

#define BINARY_OP_ACCESSOR(Name) \
    template<typename _Right> \
	BinaryOp<_Derived, _Right, functor::BinaryMathFunctor_ ## Name <Scalar>, false > Name (const MatrixBase<_Right>& rhs) const { \
		return BinaryOp<_Derived, _Right, functor::BinaryMathFunctor_ ## Name <Scalar>, false >(derived(), rhs.derived()); \
	} \
    template<typename _Right, typename T = typename std::enable_if<std::is_convertible<_Right, Scalar>::value, \
        BinaryOp<_Derived, HostScalar<Scalar>, functor::BinaryMathFunctor_ ## Name <Scalar>, false > >::type > \
    T Name(const _Right& rhs) const { \
		return BinaryOp<_Derived, HostScalar<Scalar>, functor::BinaryMathFunctor_ ## Name <Scalar>, false >(derived(), HostScalar<Scalar>(rhs)); \
	}
#define BINARY_OP_ACCESSOR_INV(Name) \
    template<typename _Left> \
        BinaryOp<_Left, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar>, false > Name ## Inv(const MatrixBase<_Left>& lhs) const { \
		return BinaryOp<_Left, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar>, false >(lhs.derived(), derived()); \
	} \
    template<typename _Left, typename T = typename std::enable_if<std::is_convertible<_Left, Scalar>::value, \
        BinaryOp<HostScalar<Scalar>, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar>, false > >::type > \
    T Name ## Inv(const _Left& lhs) const { \
		return BinaryOp<HostScalar<Scalar>, _Derived, functor::BinaryMathFunctor_ ## Name <Scalar>, false >(HostScalar<Scalar>(lhs), derived()); \
	}

/**
* \brief computes the component-wise multiplation (this*rhs)
*/
BINARY_OP_ACCESSOR(cwiseMul)

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

#undef BINARY_OP_ACCESSOR
#undef BINARY_OP_ACCESSOR_INV