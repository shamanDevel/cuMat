#ifndef __CUMAT_BINARY_OPS_H__
#define __CUMAT_BINARY_OPS_H__

#include <type_traits>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "CwiseOp.h"
#include "NumTraits.h"

CUMAT_NAMESPACE_BEGIN


namespace internal {
    template<typename _Left, typename _Right, typename _BinaryFunctor>
    struct traits<BinaryOp<_Left, _Right, _BinaryFunctor> >
    {
        using Scalar = typename _BinaryFunctor::ReturnType;
        enum
        {
            BroadcastRowsLeft = (internal::traits<_Left>::RowsAtCompileTime == 1),
            BroadcastColsLeft = (internal::traits<_Left>::ColsAtCompileTime == 1),
            BroadcastBatchesLeft = (internal::traits<_Left>::BatchesAtCompileTime == 1),

            BroadcastRowsRight = (internal::traits<_Right>::RowsAtCompileTime == 1),
            BroadcastColsRight = (internal::traits<_Right>::ColsAtCompileTime == 1),
            BroadcastBatchesRight = (internal::traits<_Right>::BatchesAtCompileTime == 1),

            FlagsLeft = internal::traits<_Left>::Flags,
            RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
            ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
            BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

            FlagsRight = internal::traits<_Right>::Flags,
            RowsRight = internal::traits<_Right>::RowsAtCompileTime,
            ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
            BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

            Flags = (BroadcastRowsRight || BroadcastColsRight) ? FlagsLeft : FlagsRight, //use the flags of the matrix type
            RowsAtCompileTime = (RowsLeft == Dynamic || RowsRight == Dynamic) ?
                Dynamic :
                (BroadcastRowsRight ? RowsLeft : RowsRight),
            ColsAtCompileTime = (ColumnsLeft == Dynamic || ColumnsRight == Dynamic) ?
                Dynamic :
                (BroadcastColsRight ? ColumnsLeft : ColumnsLeft),
            BatchesAtCompileTime = (BatchesLeft == Dynamic || BatchesRight == Dynamic) ?
                Dynamic :
                (BroadcastBatchesRight ? BatchesLeft : BatchesRight),

            AccessFlags = ReadCwise
        };
        typedef CwiseSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}

/**
* \brief A generic binary operator supporting broadcasting.
* The binary functor can be any class or structure that supports
* the following method:
* \code
* __device__ const _Scalar& operator()(const Scalar& left, const Scalar& right, Index row, Index col, Index batch);
* \endcode
* \tparam _Left the matrix on the left hand side
* \tparam _Right the matrix on the right hand side
* \tparam _BinaryFunctor the binary functor
* \tparam _IsLogic: true -> the op returns a bool matrix, false -> the same scalar is returned
*/
template<typename _Left, typename _Right, typename _BinaryFunctor>
class BinaryOp : public CwiseOp<BinaryOp<_Left, _Right, _BinaryFunctor> >
{
public:
    typedef CwiseOp<BinaryOp<_Left, _Right, _BinaryFunctor> > Base;
    typedef BinaryOp<_Left, _Right, _BinaryFunctor> Type;
    CUMAT_PUBLIC_API

    using LeftScalar = typename internal::traits<_Left>::Scalar;
	using RightScalar = typename internal::traits<_Right>::Scalar;
    enum
    {
        BroadcastRowsLeft = (internal::traits<_Left>::RowsAtCompileTime == 1),
        BroadcastColsLeft = (internal::traits<_Left>::ColsAtCompileTime == 1),
        BroadcastBatchesLeft = (internal::traits<_Left>::BatchesAtCompileTime == 1),

        BroadcastRowsRight = (internal::traits<_Right>::RowsAtCompileTime == 1),
        BroadcastColsRight = (internal::traits<_Right>::ColsAtCompileTime == 1),
        BroadcastBatchesRight = (internal::traits<_Right>::BatchesAtCompileTime == 1),

        FlagsLeft = internal::traits<_Left>::Flags,
        RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
        ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
        BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

        FlagsRight = internal::traits<_Right>::Flags,
        RowsRight = internal::traits<_Right>::RowsAtCompileTime,
        ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
        BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

        //Flags, Rows, Columns, Batches defined in CUMAT_PUBLIC_API
    };

protected:
    typedef typename MatrixReadWrapper<_Left, AccessFlags::ReadCwise>::type left_wrapped_t;
    typedef typename MatrixReadWrapper<_Right, AccessFlags::ReadCwise>::type right_wrapped_t;
    const left_wrapped_t left_;
    const right_wrapped_t right_;
    const _BinaryFunctor functor_;

public:
    BinaryOp(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right,
        const _BinaryFunctor& functor = _BinaryFunctor())
        : left_(left.derived()), right_(right.derived()), functor_(functor)
    {
        CUMAT_STATIC_ASSERT((std::is_same<typename internal::traits<_Left>::Scalar, typename internal::traits<_Right>::Scalar>::value),
            "No implicit casting is allowed in binary operations.");

        if (RowsLeft == Dynamic && RowsRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.rows() == right.rows());
        }
        CUMAT_STATIC_ASSERT(!(RowsLeft > 1 && RowsRight > 1) || (RowsLeft == RowsRight), "matrix sizes don't match");

        if (ColumnsLeft == Dynamic && ColumnsRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.cols() == right.cols());
        }
        CUMAT_STATIC_ASSERT(!(ColumnsLeft > 1 && ColumnsRight > 1) || (ColumnsLeft == ColumnsRight), "matrix sizes don't match");

        if (BatchesLeft == Dynamic && BatchesRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.batches() == right.batches());
        }
        CUMAT_STATIC_ASSERT(!(BatchesLeft > 1 && BatchesRight > 1) || (BatchesLeft == BatchesRight), "matrix sizes don't match");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const
    {
        return BroadcastRowsRight ? left_.rows() : right_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        return BroadcastColsRight ? left_.cols() : right_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return BroadcastBatchesRight ? left_.batches() : right_.batches();
    }

    __device__ CUMAT_STRONG_INLINE LeftScalar getLeft(Index row, Index col, Index batch, Index linear) const
    {
        return left_.derived().coeff(
            BroadcastRowsLeft ? 0 : row,
            BroadcastColsLeft ? 0 : col,
            BroadcastBatchesLeft ? 0 : batch,
            linear);
    }
    __device__ CUMAT_STRONG_INLINE RightScalar getRight(Index row, Index col, Index batch, Index linear) const
    {
        return right_.derived().coeff(
            BroadcastRowsRight ? 0 : row,
            BroadcastColsRight ? 0 : col,
            BroadcastBatchesRight ? 0 : batch,
            linear);
    }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index linear) const
    {
        return functor_(getLeft(row, col, batch, linear), getRight(row, col, batch, linear), row, col, batch);
    }
};


//-----------------
// Normal binary ops
//-----------------

namespace functor
{
#define DECLARE_FUNCTOR(Name) \
	template<typename _Scalar> class BinaryMathFunctor_ ## Name

#define DEFINE_GENERAL_FUNCTOR(Name, Fn) \
	template<typename _Scalar> \
	struct BinaryMathFunctor_ ## Name \
	{ \
	public: \
		typedef _Scalar ReturnType; \
		__device__ CUMAT_STRONG_INLINE _Scalar operator()(const _Scalar& x, const _Scalar& y, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

#define DEFINE_FUNCTOR(Name, Scalar, Fn) \
	template<> \
	struct BinaryMathFunctor_ ## Name <Scalar> \
	{ \
	public: \
		typedef Scalar ReturnType; \
		__device__ CUMAT_STRONG_INLINE Scalar operator()(const Scalar& x, const Scalar& y, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

	//the std::is_same hack is needed so that the static assert evaluates to false only on template instantiation
#define DEFINE_FALLBACK_FUNCTOR(Name) \
	template<typename _Scalar> \
	struct BinaryMathFunctor_ ## Name \
	{ \
	public: \
		CUMAT_STATIC_ASSERT((!std::is_same<_Scalar, _Scalar>::value), "Functor not available for the selected type"); \
	}

#define DEFINE_FUNCTOR_FLOAT(Name, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);     \
	DEFINE_FUNCTOR(Name, float, Fn);   \
	DEFINE_FUNCTOR(Name, double, Fn);
#define DEFINE_FUNCTOR_FLOAT_COMPLEX(Name, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);     \
	DEFINE_FUNCTOR(Name, float, Fn);   \
	DEFINE_FUNCTOR(Name, double, Fn);  \
    DEFINE_FUNCTOR(Name, cfloat, Fn);  \
	DEFINE_FUNCTOR(Name, cdouble, Fn);
#define DEFINE_FUNCTOR_INT(Name, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);   \
	DEFINE_FUNCTOR(Name, int, Fn);   \
	DEFINE_FUNCTOR(Name, long, Fn);
#define DEFINE_FUNCTOR_SINGLE(Name, Type, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);     \
	DEFINE_FUNCTOR(Name, Type, Fn);

    DEFINE_GENERAL_FUNCTOR(cwiseAdd, x + y);
    DEFINE_GENERAL_FUNCTOR(cwiseSub, x - y);
    DEFINE_GENERAL_FUNCTOR(cwiseMul, x * y);
	DEFINE_GENERAL_FUNCTOR(cwiseDot, x * y);
    DEFINE_GENERAL_FUNCTOR(cwiseDiv, x / y);
    DEFINE_FUNCTOR_INT(cwiseMod, x % y);
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwisePow, pow(x, y));

	DEFINE_FUNCTOR_INT(cwiseBinaryAnd, x & y);
	DEFINE_FUNCTOR_INT(cwiseBinaryOr, x | y);
	DEFINE_FUNCTOR_INT(cwiseBinaryXor, x ^ y);
	DEFINE_FUNCTOR_SINGLE(cwiseLogicalAnd, bool, x && y);
	DEFINE_FUNCTOR_SINGLE(cwiseLogicalOr, bool, x || y);
	DEFINE_FUNCTOR_SINGLE(cwiseLogicalXor, bool, x ^ y);

#undef DECLARE_FUNCTOR
#undef DEFINE_GENERAL_FUNCTOR
#undef DEFINE_FUNCTOR
#undef DEFINE_FUNCTOR_FLOAT
#undef DEFINE_FUNCTOR_FLOAT_COMPLEX
#undef DEFINE_FUNCTOR_INT
#undef DEFINE_FALLBACK_FUNCTOR
#undef DEFINE_FUNCTOR_SINGLE
} //end namespace functor

CUMAT_NAMESPACE_END

#define BINARY_OP_SCALAR(Name, Op) \
    template<typename _Left, typename _Right, \
        typename S = typename CUMAT_NAMESPACE internal::traits<_Right>::Scalar, \
        typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<_Left, S>::value, \
            CUMAT_NAMESPACE BinaryOp<CUMAT_NAMESPACE HostScalar<S>, _Right, CUMAT_NAMESPACE Op<S>> >::type>\
    T Name(const _Left& left, const CUMAT_NAMESPACE MatrixBase<_Right>& right) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(binaryOp)  \
        return CUMAT_NAMESPACE BinaryOp<CUMAT_NAMESPACE HostScalar<S>, _Right, CUMAT_NAMESPACE Op<S>>(CUMAT_NAMESPACE make_host_scalar<S>(left), right); \
    } \
    template<typename _Left, typename _Right, \
        typename S = typename CUMAT_NAMESPACE internal::traits<_Left>::Scalar, \
        typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<_Right, S>::value, \
            CUMAT_NAMESPACE BinaryOp<_Left, CUMAT_NAMESPACE HostScalar<S>, CUMAT_NAMESPACE Op<S>> >::type>\
    T Name(const CUMAT_NAMESPACE MatrixBase<_Left>& left, const _Right& right) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(binaryOp)  \
        return CUMAT_NAMESPACE BinaryOp<_Left, CUMAT_NAMESPACE HostScalar<S>, CUMAT_NAMESPACE Op<S>>(left, CUMAT_NAMESPACE make_host_scalar<S>(right)); \
    }
#define BINARY_OP(Name, Op) \
    template<typename _Left, typename _Right> \
    CUMAT_NAMESPACE BinaryOp<_Left, _Right, CUMAT_NAMESPACE Op<typename CUMAT_NAMESPACE internal::traits<_Left>::Scalar>> \
    Name(const CUMAT_NAMESPACE MatrixBase<_Left>& left, const CUMAT_NAMESPACE MatrixBase<_Right>& right) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(binaryOp)  \
        return CUMAT_NAMESPACE BinaryOp<_Left, _Right, CUMAT_NAMESPACE Op<typename CUMAT_NAMESPACE internal::traits<_Left>::Scalar>>(left, right); \
    } \
    BINARY_OP_SCALAR(Name, Op)

//Operator overloading
CUMAT_NAMESPACE_BEGIN
    BINARY_OP(operator+, functor::BinaryMathFunctor_cwiseAdd)
    BINARY_OP(operator-, functor::BinaryMathFunctor_cwiseSub)
    BINARY_OP(operator%, functor::BinaryMathFunctor_cwiseMod)
    BINARY_OP_SCALAR(operator*, functor::BinaryMathFunctor_cwiseMul)
    BINARY_OP_SCALAR(operator/, functor::BinaryMathFunctor_cwiseDiv)
CUMAT_NAMESPACE_END

//Global binary functions
CUMAT_FUNCTION_NAMESPACE_BEGIN

/**
 * \brief Computes the component-wise exponent pow(left, right).
 * \tparam _Left the left matrix type
 * \tparam _Right the right matrix type
 * \param left the left matrix
 * \param right the right matrix
 * \return The expression computing the cwise exponent
 */
BINARY_OP(pow, functor::BinaryMathFunctor_cwisePow)

CUMAT_FUNCTION_NAMESPACE_END

#undef BINARY_OP
#undef BINARY_OP_SCALAR

//-----------------
// Logical binary ops
//-----------------

CUMAT_NAMESPACE_BEGIN

namespace functor
{
#define DECLARE_FUNCTOR(Name) \
	template<typename _Scalar> class BinaryLogicFunctor_ ## Name

#define DEFINE_GENERAL_FUNCTOR(Name, Fn) \
	template<typename _Scalar> \
	struct BinaryLogicFunctor_ ## Name \
	{ \
	public: \
		typedef bool ReturnType; \
		__device__ CUMAT_STRONG_INLINE bool operator()(const _Scalar& x, const _Scalar& y, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

    DEFINE_GENERAL_FUNCTOR(cwiseEqual, x == y);
    DEFINE_GENERAL_FUNCTOR(cwiseNequal, x != y);
    DEFINE_GENERAL_FUNCTOR(cwiseLess, x < y);
    DEFINE_GENERAL_FUNCTOR(cwiseGreater, x > y);
    DEFINE_GENERAL_FUNCTOR(cwiseLessEq, x <= y);
    DEFINE_GENERAL_FUNCTOR(cwiseGreaterEq, x >= y);

#undef DECLARE_FUNCTOR
#undef DEFINE_GENERAL_FUNCTOR
#undef DEFINE_FUNCTOR
#undef DEFINE_FUNCTOR_FLOAT
#undef DEFINE_FUNCTOR_INT
} //end namespace functor

CUMAT_NAMESPACE_END

#define BINARY_OP_SCALAR(Name, Op) \
    template<typename _Left, typename _Right, \
        typename S = typename CUMAT_NAMESPACE internal::traits<_Right>::Scalar, \
        typename T = typename std::enable_if<std::is_convertible<_Left, S>::value, \
            CUMAT_NAMESPACE BinaryOp<CUMAT_NAMESPACE HostScalar<S>, _Right, CUMAT_NAMESPACE Op<S>> >::type>\
    T Name(const _Left& left, const CUMAT_NAMESPACE MatrixBase<_Right>& right) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(logicOp)  \
        return CUMAT_NAMESPACE BinaryOp<CUMAT_NAMESPACE HostScalar<S>, _Right, CUMAT_NAMESPACE Op<S>>(CUMAT_NAMESPACE make_host_scalar<S>(S(left)), right); \
    } \
    template<typename _Left, typename _Right, \
        typename S = typename CUMAT_NAMESPACE internal::traits<_Left>::Scalar, \
        typename T = typename std::enable_if<std::is_convertible<_Right, S>::value, \
            CUMAT_NAMESPACE BinaryOp<_Left, CUMAT_NAMESPACE HostScalar<S>, CUMAT_NAMESPACE Op<S>> >::type>\
    T Name(const CUMAT_NAMESPACE MatrixBase<_Left>& left, const _Right& right) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(logicOp)  \
        return CUMAT_NAMESPACE BinaryOp<_Left, CUMAT_NAMESPACE HostScalar<S>, CUMAT_NAMESPACE Op<S>>(left, CUMAT_NAMESPACE make_host_scalar<S>(S(right))); \
    }
#define BINARY_OP(Name, Op) \
    template<typename _Left, typename _Right> \
    CUMAT_NAMESPACE BinaryOp<_Left, _Right, CUMAT_NAMESPACE Op<typename CUMAT_NAMESPACE internal::traits<_Left>::Scalar>> \
    Name(const CUMAT_NAMESPACE MatrixBase<_Left>& left, const CUMAT_NAMESPACE MatrixBase<_Right>& right) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(logicOp)  \
        return CUMAT_NAMESPACE BinaryOp<_Left, _Right, CUMAT_NAMESPACE Op<typename CUMAT_NAMESPACE internal::traits<_Left>::Scalar>>(left, right); \
    } \
    BINARY_OP_SCALAR(Name, Op)

//Operator overloading
CUMAT_NAMESPACE_BEGIN
BINARY_OP(operator==, functor::BinaryLogicFunctor_cwiseEqual)
BINARY_OP(operator!=, functor::BinaryLogicFunctor_cwiseNequal)
BINARY_OP(operator<, functor::BinaryLogicFunctor_cwiseLess)
BINARY_OP(operator>, functor::BinaryLogicFunctor_cwiseGreater)
BINARY_OP(operator<=, functor::BinaryLogicFunctor_cwiseLessEq)
BINARY_OP(operator>=, functor::BinaryLogicFunctor_cwiseGreaterEq)
CUMAT_NAMESPACE_END

#undef BINARY_OP
#undef BINARY_OP_SCALAR

#endif
