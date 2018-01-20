#ifndef __CUMAT_UNARY_OPS_H__
#define __CUMAT_UNARY_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "CwiseOp.h"

CUMAT_NAMESPACE_BEGIN

namespace internal {
	template<typename _Child, typename _UnaryFunctor>
	struct traits<UnaryOp<_Child, _UnaryFunctor> >
	{
		using Scalar = typename internal::traits<_Child>::Scalar;
		enum
		{
			Flags = internal::traits<_Child>::Flags,
			RowsAtCompileTime = internal::traits<_Child>::RowsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Child>::ColsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise
		};
	};
}

/**
* \brief A generic unary operator.
* The unary functor can be any class or structure that supports
* the following method:
* \code
* __device__ const _Scalar& operator()(const Scalar& value, Index row, Index col, Index batch);
* \endcode
* \tparam _Child the matrix that is transformed by this unary function
* \tparam _UnaryFunctor the transformation functor
*/
template<typename _Child, typename _UnaryFunctor>
class UnaryOp : public CwiseOp<UnaryOp<_Child, _UnaryFunctor> >
{
public:
	typedef CwiseOp<UnaryOp<_Child, _UnaryFunctor> > Base;
	using Scalar = typename internal::traits<_Child>::Scalar;
	enum
	{
		Flags = internal::traits<_Child>::Flags,
		Rows = internal::traits<_Child>::RowsAtCompileTime,
		Columns = internal::traits<_Child>::ColsAtCompileTime,
		Batches = internal::traits<_Child>::BatchesAtCompileTime
	};

protected:
    typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
	const child_wrapped_t child_;
	const _UnaryFunctor functor_;

public:
	explicit UnaryOp(const MatrixBase<_Child>& child, const _UnaryFunctor& functor = _UnaryFunctor())
		: child_(child.derived()), functor_(functor)
	{}

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return child_.rows(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return child_.cols(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

	__device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch) const
	{
		return functor_(child_.derived().coeff(row, col, batch), row, col, batch);
	}
};

// GENERAL UNARY OPERATIONS
namespace functor
{
#define DECLARE_FUNCTOR(Name) \
	template<typename _Scalar> class UnaryMathFunctor_ ## Name

#define DEFINE_GENERAL_FUNCTOR(Name, Fn) \
	template<typename _Scalar> \
	class UnaryMathFunctor_ ## Name \
	{ \
	public: \
		__device__ CUMAT_STRONG_INLINE _Scalar operator()(const _Scalar& x, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

#define DEFINE_FUNCTOR(Name, Scalar, Fn) \
	template<> \
	class UnaryMathFunctor_ ## Name <Scalar> \
	{ \
	public: \
		__device__ CUMAT_STRONG_INLINE Scalar operator()(const Scalar& x, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

#define DEFINE_FUNCTOR_FLOAT(Name, Fn) \
	DEFINE_FUNCTOR(Name, float, Fn);   \
	DEFINE_FUNCTOR(Name, double, Fn);
#define DEFINE_FUNCTOR_INT(Name, Fn) \
	DEFINE_FUNCTOR(Name, int, Fn);   \
	DEFINE_FUNCTOR(Name, long, Fn);

	//DECLARE_FUNCTOR(negate); //this is already done in ForwardDeclarations.h

	DEFINE_GENERAL_FUNCTOR(cwiseNegate, (-x));
	DEFINE_GENERAL_FUNCTOR(cwiseAbs, abs(x));
    DEFINE_GENERAL_FUNCTOR(cwiseAbs2, x*x);
	DEFINE_GENERAL_FUNCTOR(cwiseInverse, 1/x);

	DEFINE_FUNCTOR_FLOAT(cwiseExp, exp(x));
	DEFINE_FUNCTOR_FLOAT(cwiseLog, log(x));
	DEFINE_FUNCTOR_FLOAT(cwiseLog1p, log1p(x));
	DEFINE_FUNCTOR_FLOAT(cwiseLog10, log10(x));

	DEFINE_FUNCTOR_FLOAT(cwiseSqrt, sqrt(x));
	DEFINE_FUNCTOR_FLOAT(cwiseRsqrt, rsqrt(x));
	DEFINE_FUNCTOR_FLOAT(cwiseCbrt, cbrt(x));
	DEFINE_FUNCTOR_FLOAT(cwiseRcbrt, rcbrt(x));
	
	DEFINE_FUNCTOR_FLOAT(cwiseSin, sin(x));
	DEFINE_FUNCTOR_FLOAT(cwiseCos, cos(x));
	DEFINE_FUNCTOR_FLOAT(cwiseTan, tan(x));
	DEFINE_FUNCTOR_FLOAT(cwiseAsin, asin(x));
	DEFINE_FUNCTOR_FLOAT(cwiseAcos, acos(x));
	DEFINE_FUNCTOR_FLOAT(cwiseAtan, atan(x));
	DEFINE_FUNCTOR_FLOAT(cwiseSinh, sinh(x));
	DEFINE_FUNCTOR_FLOAT(cwiseCosh, cosh(x));
	DEFINE_FUNCTOR_FLOAT(cwiseTanh, tanh(x));
	DEFINE_FUNCTOR_FLOAT(cwiseAsinh, asinh(x));
	DEFINE_FUNCTOR_FLOAT(cwiseAcosh, acosh(x));
	DEFINE_FUNCTOR_FLOAT(cwiseAtanh, atanh(x));

	DEFINE_FUNCTOR_FLOAT(cwiseFloor, floor(x));
	DEFINE_FUNCTOR_FLOAT(cwiseCeil, ceil(x));
	DEFINE_FUNCTOR_FLOAT(cwiseRound, round(x));
	DEFINE_FUNCTOR_INT(cwiseFloor, x);
	DEFINE_FUNCTOR_INT(cwiseCeil, x);
	DEFINE_FUNCTOR_INT(cwiseRound, x);

	DEFINE_FUNCTOR_FLOAT(cwiseErf, erf(x));
	DEFINE_FUNCTOR_FLOAT(cwiseErfc, erfc(x));
	DEFINE_FUNCTOR_FLOAT(cwiseLgamma, lgamma(x));

#undef DECLARE_FUNCTOR
#undef DEFINE_GENERAL_FUNCTOR
#undef DEFINE_FUNCTOR
#undef DEFINE_FUNCTOR_FLOAT
#undef DEFINE_FUNCTOR_INT
} //end namespace functor


// CASTING

namespace functor {
	/**
	 * \brief Cast a scalar of type _Source to a scalar of type _Target.
	 * The functor provides a function <code>static __device__ _Target cast(_Source source)</code>
	 * for casting.
	 * \tparam _Source the source type 
	 * \tparam _Target the target type
	 */
	template<typename _Source, typename _Target>
	struct CastFunctor
	{
		static __device__ CUMAT_STRONG_INLINE _Target cast(const _Source& source)
		{
			//general implementation
			return _Target(source);
		}
	};
	//TODO: specializations for complex
} //end namespace functor

namespace internal {
	template<typename _Child, typename _Target>
	struct traits<CastingOp<_Child, _Target> >
	{
		using Scalar = _Target;
		enum
		{
			Flags = internal::traits<_Child>::Flags,
			RowsAtCompileTime = internal::traits<_Child>::RowsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Child>::ColsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime
		};
	};
}

/**
 * \brief Casting operator from the type of the matrix _Child to the datatype _Target.
 * It uses functor::CastFunctor for the casting
 * \tparam _Child the child expression
 * \tparam _Target the target type
 */
template<typename _Child, typename _Target>
class CastingOp : public CwiseOp<CastingOp<_Child, _Target> >
{
public:
	typedef CwiseOp<CastingOp<_Child, _Target> > Base;
	using SourceType = typename internal::traits<_Child>::Scalar;
	using TargetType = _Target;
	using Scalar = TargetType;
	enum
	{
		Flags = internal::traits<_Child>::Flags,
		Rows = internal::traits<_Child>::RowsAtCompileTime,
		Columns = internal::traits<_Child>::ColsAtCompileTime,
		Batches = internal::traits<_Child>::BatchesAtCompileTime
	};

protected:
	const _Child child_;

public:
	explicit CastingOp(const MatrixBase<_Child>& child)
		: child_(child.derived())
	{}

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return child_.rows(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return child_.cols(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

	__device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch) const
	{
		return functor::CastFunctor<SourceType, TargetType>::cast(child_.derived().coeff(row, col, batch));
	}
};


// diagonal() and asDiagonal()

namespace internal {
    template<typename _Child>
    struct traits<AsDiagonalOp<_Child> >
    {
        using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            Size = internal::traits<_Child>::RowsAtCompileTime == 1 ? internal::traits<_Child>::ColsAtCompileTime : internal::traits<_Child>::RowsAtCompileTime,
            RowsAtCompileTime = Size,
            ColsAtCompileTime = Size,
            BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime
        };
    };
}
/**
* \brief A wrapper operation that expresses the compile-time vector as a diagonal matrix.
* This is the return type of \c asDiagonal()
* \tparam _Child the child expression
*/
template<typename _Child>
class AsDiagonalOp : public CwiseOp<AsDiagonalOp<_Child> >
{
public:
    typedef CwiseOp<AsDiagonalOp<_Child> > Base;
    using Scalar = typename internal::traits<_Child>::Scalar;
    enum
    {
        Flags = internal::traits<_Child>::Flags,
        Size = internal::traits<_Child>::RowsAtCompileTime == 1 ? internal::traits<_Child>::ColsAtCompileTime : internal::traits<_Child>::RowsAtCompileTime,
        IsRowVector = internal::traits<_Child>::RowsAtCompileTime == 1,
        Rows = Size,
        Columns = Size,
        Batches = internal::traits<_Child>::BatchesAtCompileTime
    };

protected:
    const _Child child_;
    const Index size_;

public:
    explicit AsDiagonalOp(const MatrixBase<_Child>& child)
        : child_(child.derived())
        , size_(child.rows()==1 ? child.cols() : child.rows())
    {
        CUMAT_STATIC_ASSERT(internal::traits<_Child>::RowsAtCompileTime == 1 || internal::traits<_Child>::ColsAtCompileTime == 1,
            "The child expression must be a compile-time row or column vector");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return size_; }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return size_; }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch) const
    {
        if (row == col)
        {
            if (IsRowVector)
                return child_.derived().coeff(0, col, batch);
            else
                return child_.derived().coeff(row, 0, batch);
        } else
        {
            return Scalar(0);
        }
    }
};


namespace internal {
    template<typename _Child>
    struct traits<ExtractDiagonalOp<_Child> >
    {
        using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            RowsAtCompileTime = (internal::traits<_Child>::RowsAtCompileTime == Dynamic || internal::traits<_Child>::RowsAtCompileTime == Dynamic)
                ? Dynamic
                : (internal::traits<_Child>::ColsAtCompileTime < internal::traits<_Child>::RowsAtCompileTime
                    ? internal::traits<_Child>::ColsAtCompileTime
                    : internal::traits<_Child>::RowsAtCompileTime),
            ColsAtCompileTime = 1,
            BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime
        };
    };
}
/**
* \brief The operation that extracts the main diagonal of a matrix and returns it as a column vector.
* The matrix must not necessarily be square
* This is the return type of \c diagonal()
* \tparam _Child the child expression
*/
template<typename _Child>
class ExtractDiagonalOp : public CwiseOp<ExtractDiagonalOp<_Child> >
{
public:
    typedef CwiseOp<ExtractDiagonalOp<_Child> > Base;
    using Scalar = typename internal::traits<_Child>::Scalar;
    enum
    {
        Flags = internal::traits<_Child>::Flags,
        Rows = (internal::traits<_Child>::RowsAtCompileTime == Dynamic || internal::traits<_Child>::RowsAtCompileTime == Dynamic)
            ? Dynamic 
            : (internal::traits<_Child>::ColsAtCompileTime < internal::traits<_Child>::RowsAtCompileTime
                ? internal::traits<_Child>::ColsAtCompileTime
                : internal::traits<_Child>::RowsAtCompileTime),
        Columns = 1,
        Batches = internal::traits<_Child>::BatchesAtCompileTime
    };

protected:
    const _Child child_;
    const Index size_;

public:
    explicit ExtractDiagonalOp(const MatrixBase<_Child>& child)
        : child_(child.derived())
        , size_(child.rows() < child.cols() ? child.rows() : child.cols())
    {}

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return size_; }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return 1; }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch) const
    {
        return child_.derived().coeff(row, row, batch);
    }
};


CUMAT_NAMESPACE_END


//Global binary functions
CUMAT_FUNCTION_NAMESPACE_BEGIN

#define UNARY_OP(Name, Op) \
    template<typename _Derived> \
    CUMAT_NAMESPACE UnaryOp<_Derived, CUMAT_NAMESPACE functor::UnaryMathFunctor_ ## Op <typename CUMAT_NAMESPACE internal::traits<_Derived>::Scalar>> \
    Name(const CUMAT_NAMESPACE MatrixBase<_Derived>& mat) \
    { \
        return CUMAT_NAMESPACE UnaryOp<_Derived, CUMAT_NAMESPACE functor::UnaryMathFunctor_ ## Op <typename CUMAT_NAMESPACE internal::traits<_Derived>::Scalar>>(mat.derived()); \
    }

UNARY_OP(abs, cwiseAbs);
UNARY_OP(inverse, cwiseInverse);
UNARY_OP(floor, cwiseFloor);
UNARY_OP(ceil, cwiseCeil);
UNARY_OP(round, cwiseRound);

UNARY_OP(exp, cwiseExp);
UNARY_OP(log, cwiseLog);
UNARY_OP(log1p, cwiseLog1p);
UNARY_OP(log10, cwiseLog10);
UNARY_OP(sqrt, cwiseSqrt);
UNARY_OP(rsqrt, cwiseRsqrt);
UNARY_OP(cbrt, cwiseCbrt);
UNARY_OP(rcbrt, cwiseRcbrt);

UNARY_OP(sin, cwiseSin);
UNARY_OP(cos, cwiseCos);
UNARY_OP(tan, cwiseTan);
UNARY_OP(asin, cwiseAsin);
UNARY_OP(acos, cwiseAcos);
UNARY_OP(atan, cwiseAtan);
UNARY_OP(sinh, cwiseSinh);
UNARY_OP(cosh, cwiseCosh);
UNARY_OP(tanh, cwiseTanh);
UNARY_OP(asinh, cwiseAsinh);
UNARY_OP(acosh, cwiseAcosh);
UNARY_OP(atanh, cwiseAtanh);

UNARY_OP(erf, cwiseErf);
UNARY_OP(erfc, cwiseErfc);
UNARY_OP(lgamma, cwiseLgamma);

#undef UNARY_OP

CUMAT_FUNCTION_NAMESPACE_END


#endif
