//Included inside MatrixBase, define the accessors

#define UNARY_OP_ACCESSOR(Name) \
	UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> > Name () { \
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
* \brief computes the component-wise inverse (x -> 1/x)
*/
UNARY_OP_ACCESSOR(cwiseInverse);

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

UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate<Scalar> > operator-() {
	return UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate <Scalar> >(derived());
}

#undef UNARY_OP_ACCESSOR