//Included inside MatrixBase, define the accessors

#define UNARY_OP_ACCESSOR(Name) \
	UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> > Name () { \
		return UnaryOp<_Derived, functor::UnaryMathFunctor_ ## Name <Scalar> >(derived()); \
	}

UNARY_OP_ACCESSOR(negate);

#undef UNARY_OP_ACCESSOR