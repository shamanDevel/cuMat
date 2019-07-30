#ifndef __CUMAT_FORWARD_DECLARATIONS_H__
#define __CUMAT_FORWARD_DECLARATIONS_H__

#include "Macros.h"
#include "Constants.h"
#include <thrust/complex.h>

CUMAT_NAMESPACE_BEGIN

/**
* \brief complex float type
*/
typedef thrust::complex<float> cfloat;
/**
* \brief complex double type
*/
typedef thrust::complex<double> cdouble;

/**
* \brief The datatype used for matrix indexing
*/
typedef ptrdiff_t Index;

/**
 * \brief Type used as a error message to indicate ops that are only supported in .cu files
 */
struct THIS_FUNCTION_REQUIRES_THE_FILE_TO_BE_COMPILED_WITH_NVCC;

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
	* typedef ... SrcTag; //The source tag for the assignment dispatch
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

    /**
     * \brief General assignment dispatcher.
     * Implementations must have a \code static void assign(_Dst& dst, const _Src& src) \endcode function.
     * \tparam _Dst 
     * \tparam _Src 
     * \tparam _AssignmentMode 
     * \tparam _DstTag
     * \tparam _SrcTag
     */
    template<typename _Dst, typename _Src, AssignmentMode _AssignmentMode, typename _DstTag, typename _SrcTag>
    struct Assignment;

    struct CwiseSrcTag {};
    /**
     * \brief "Dense" destination, in the sense that there as a simple mapping from a 
     * linear index (max index returned by \c size() ) to the row, column and batch index.
     * Must follow the assignment mode CwiseWrite.
     */
    struct DenseDstTag {};
    struct SparseDstTag{};
    typedef void DeletedSrcTag;
    typedef void DeletedDstTag;
}

template<typename _Derived> class MatrixBase;

template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags> class Matrix;
template<typename _Scalar, int _Batches, int _Flags> class SparseMatrix;
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _MatrixType> class MatrixBlock;
namespace internal {
    template <typename _MatrixType> class MatrixInplaceAssignment;
    template <typename _MatrixType> class SparseMatrixInplaceAssignment;
    template <typename _MatrixType> class SparseMatrixDirectAccess;
}

template<typename _Derived> class CwiseOp;
template<typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _NullaryFunctor> class NullaryOp;
template<typename _Child, typename _UnaryFunctor> class UnaryOp;
template<typename _Left, typename _Right, typename _BinaryFunctor> class BinaryOp;
template<typename _Child, typename _Target> class CastingOp;
template<typename _Derived, bool _Conjugated> class TransposeOp;
template<typename _Child, typename _ReductionOp, typename _Algorithm> class ReductionOp_DynamicSwitched;
template<typename _Child, typename _ReductionOp, int _Axis, typename _Algorithm> class ReductionOp_StaticSwitched;

namespace internal { 
    enum class ProductArgOp; 
    template<typename _LeftScalar, typename _RightScalar, ProductArgOp _OpLeft, ProductArgOp _OpRight, ProductArgOp _OpOutput> struct ProductElementFunctor;
}
template<typename _Left, typename _Right, internal::ProductArgOp _OpLeft, internal::ProductArgOp _OpRight, internal::ProductArgOp _OpOutput> class ProductOp;

template<typename _Child> class AsDiagonalOp;
template<typename _Child> class ExtractDiagonalOp;
template <typename _Child, bool _Imag, bool _Lvalue> class ExtractComplexPartOp;
template<typename _Child, Axis _Row, Axis _Col, Axis _Batch> class SwapAxisOp;

namespace functor
{
	//component-wise functors
    //nullary
    template<typename _Scalar> struct ConstantFunctor;
    template<typename _Scalar> struct IdentityFunctor;
    //unary
	template<typename _Scalar> struct UnaryMathFunctor_cwiseNegate;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAbs;
    template<typename _Scalar> struct UnaryMathFunctor_cwiseAbs2;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseInverse;
    template<typename _Scalar> struct UnaryMathFunctor_cwiseInverseCheck;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseExp;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseLog;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseLog1p;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseLog10;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseSqrt;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseRsqrt;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseCbrt;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseRcbrt;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseSin;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseCos;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseTan;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAsin;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAcos;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAtan;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseSinh;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseCosh;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseTanh;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAsinh;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAcosh;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseAtanh;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseCeil;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseFloor;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseRound;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseErf;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseErfc;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseLgamma;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseLogicalNot;
	template<typename _Scalar> struct UnaryMathFunctor_cwiseBinaryNot;
    template<typename _Scalar> struct UnaryMathFunctor_conjugate;
	//casting
	template<typename _Source, typename _Target> struct CastFunctor;
    //binary
    template<typename _Scalar> struct BinaryMathFunctor_cwiseAdd;
    template<typename _Scalar> struct BinaryMathFunctor_cwiseSub;
    template<typename _Scalar> struct BinaryMathFunctor_cwiseMul;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseDot; ///< cwise dot-product of the two arguments. Only differs from cwiseMul on vector types
    template<typename _Scalar> struct BinaryMathFunctor_cwiseDiv;
    template<typename _Scalar> struct BinaryMathFunctor_cwiseMod;
    template<typename _Scalar> struct BinaryMathFunctor_cwisePow;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseBinaryAnd;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseBinaryOr;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseBinaryXor;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseLogicalAnd;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseLogicalOr;
	template<typename _Scalar> struct BinaryMathFunctor_cwiseLogicalXor;
    template<typename _Scalar> struct BinaryLogicFunctor_cwiseEqual;
    template<typename _Scalar> struct BinaryLogicFunctor_cwiseNequal;
    template<typename _Scalar> struct BinaryLogicFunctor_cwiseLess;
    template<typename _Scalar> struct BinaryLogicFunctor_cwiseGreater;
    template<typename _Scalar> struct BinaryLogicFunctor_cwiseLessEq;
    template<typename _Scalar> struct BinaryLogicFunctor_cwiseGreaterEq;
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

// DENSE

template<typename _Solver, typename _RHS> class SolveOp;
template<typename _SolverImpl> class SolverBase;
template<typename _DecompositionImpl> class DecompositionBase;
template<typename _MatrixType> class LUDecomposition;
template<typename _MatrixType> class CholeskyDecomposition;
template<typename _Solver, typename _RHS> class SolveOp;
template<typename _Child> class DeterminantOp;
template<typename _Child> class InverseOp;
template<typename Derived, int Dims, typename InverseType, typename DetType> struct ComputeInverseWithDet;

// SPARSE

template<int _SparseFlags> struct SparsityPattern;
template<typename _Derived> class SparseMatrixBase;
template<typename _Scalar, int _Batches, int _SparseFlags> class SparseMatrix;
template<typename _Child, int _SparseFlags> class SparseExpressionOp;

// ITERATIVE LINEAR SOLVER

template<typename _Solver, typename _RHS, typename _Guess> class SolveWithGuessOp;
template<typename _SolverImpl> class IterativeSolverBase;
template<typename _MatrixType, typename _Preconditioner> class ConjugateGradient;
template<typename _MatrixType> class DiagonalPreconditioner;
template<typename _MatrixType> class IdentityPreconditioner;

CUMAT_NAMESPACE_END

#endif
