//Included inside MatrixBase, define the accessors

#define UNARY_OP_ACCESSOR(Name) \
	UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> > Name () { \
		return UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> >(derived()); \
	}

UNARY_OP_ACCESSOR(cwiseNegate);
UNARY_OP_ACCESSOR(cwiseAbs);
UNARY_OP_ACCESSOR(cwiseInverse);

UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate<Scalar> > operator-() {
	return UnaryOp<_Derived, functor::UnaryMathFunctor_cwiseNegate <Scalar> >(derived());
}

#undef UNARY_OP_ACCESSOR