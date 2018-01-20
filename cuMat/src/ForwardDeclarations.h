#ifndef __CUMAT_FORWARD_DECLARATIONS_H__
#define __CUMAT_FORWARD_DECLARATIONS_H__

#include "Macros.h"
#include <cuComplex.h>

CUMAT_NAMESPACE_BEGIN

/**
* \brief complex float type
*/
typedef cuFloatComplex cfloat;
/**
* \brief complex double type
*/
typedef cuDoubleComplex cdouble;

/**
* \brief The datatype used for matrix indexing
*/
typedef ptrdiff_t Index;

//Declares, not defines all types

namespace internal {
	/**
	* \brief Each class that inherits MatrixBase must define a specialization of internal::traits
	* that define the following:
	*
	* \code
	* typedef ... Scalar;
	* enum {
	*	Flags = ...; //storage order, see \ref Flags
	*	RowsAtCompileTime = ...;
	*	ColsAtCompileTime = ...;
	*	BatchesAtCompileTime = ...;
	*	AccessFlags = ...; //access flags, see \ref AccessFlags
	* };
	* \endcode
	*
	* \tparam T
	*/
	template<typename T> struct traits;

	// here we say once and for all that traits<const T> == traits<T>
	// When constness must affect traits, it has to be constness on template parameters on which T itself depends.
	// For example, traits<Map<const T> > != traits<Map<T> >, but
	//              traits<const Map<T> > == traits<Map<T> >
	// DIRECTLY TAKEN FROM EIGEN
	template<typename T> struct traits<const T> : traits<T> {};
}

template<typename _Derived> class MatrixBase;

template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags> class Matrix;
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _MatrixType> class MatrixBlock;

template<typename _Derived> class CwiseOp;
template<typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _NullaryFunctor> class NullaryOp;
template<typename _Child, typename _UnaryFunctor> class UnaryOp;
template<typename _Left, typename _Right, typename _BinaryFunctor, bool _IsLogic> class BinaryOp;
template<typename _Child, typename _Target> class CastingOp;
template<typename _Derived> class TransposeOp;
template<typename _Child, typename _ReductionOp> class ReductionOp_DynamicSwitched;
template<typename _Child, typename _ReductionOp, int _Axis> class ReductionOp_StaticSwitched;
template<typename _Left, typename _Right, bool _TransposedLeft, bool _TransposedRight, bool _TransposedOutput> class MultOp;
template<typename _Child> class AsDiagonalOp;
template<typename _Child> class ExtractDiagonalOp;

namespace functor
{
	//component-wise functors
    //nullary
    template<typename _Scalar> class ConstantFunctor;
    template<typename _Scalar> class IdentityFunctor;
    //unary
	template<typename _Scalar> class UnaryMathFunctor_cwiseNegate;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAbs;
	template<typename _Scalar> class UnaryMathFunctor_cwiseInverse;
	template<typename _Scalar> class UnaryMathFunctor_cwiseExp;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLog;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLog1p;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLog10;
	template<typename _Scalar> class UnaryMathFunctor_cwiseSqrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseRsqrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCbrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseRcbrt;
	template<typename _Scalar> class UnaryMathFunctor_cwiseSin;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCos;
	template<typename _Scalar> class UnaryMathFunctor_cwiseTan;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAsin;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAcos;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAtan;
	template<typename _Scalar> class UnaryMathFunctor_cwiseSinh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCosh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseTanh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAsinh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAcosh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseAtanh;
	template<typename _Scalar> class UnaryMathFunctor_cwiseCeil;
	template<typename _Scalar> class UnaryMathFunctor_cwiseFloor;
	template<typename _Scalar> class UnaryMathFunctor_cwiseRound;
	template<typename _Scalar> class UnaryMathFunctor_cwiseErf;
	template<typename _Scalar> class UnaryMathFunctor_cwiseErfc;
	template<typename _Scalar> class UnaryMathFunctor_cwiseLgamma;
	//casting
	template<typename _Source, typename _Target> struct CastFunctor;
    //binary
    template<typename _Scalar> class BinaryMathFunctor_cwiseAdd;
    template<typename _Scalar> class BinaryMathFunctor_cwiseSub;
    template<typename _Scalar> class BinaryMathFunctor_cwiseMul;
    template<typename _Scalar> class BinaryMathFunctor_cwiseDiv;
    template<typename _Scalar> class BinaryMathFunctor_cwiseMod;
    template<typename _Scalar> class BinaryMathFunctor_cwisePow;
    template<typename _Scalar> class BinaryLogicFunctor_cwiseEqual;
    template<typename _Scalar> class BinaryLogicFunctor_cwiseNequal;
    template<typename _Scalar> class BinaryLogicFunctor_cwiseLess;
    template<typename _Scalar> class BinaryLogicFunctor_cwiseGreater;
    template<typename _Scalar> class BinaryLogicFunctor_cwiseLessEq;
    template<typename _Scalar> class BinaryLogicFunctor_cwiseGreaterEq;
    //for reductions
    template<typename _Scalar> struct Sum;
    template<typename _Scalar> struct Prod;
    template<typename _Scalar> struct Min;
    template<typename _Scalar> struct Max;
    template<typename _Scalar> struct LogicalAnd;
    template<typename _Scalar> struct LogicalOr;
    template<typename _Scalar> struct BitwiseAnd;
    template<typename _Scalar> struct BitwiseOr;
}

//other typedefs
template<typename _Scalar>
using HostScalar = NullaryOp<_Scalar, 1, 1, 1, 0, functor::ConstantFunctor<_Scalar> >;

CUMAT_NAMESPACE_END

#endif