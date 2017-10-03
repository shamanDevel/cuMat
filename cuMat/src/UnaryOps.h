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
			BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime
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
	const _Child child_;
	const _UnaryFunctor functor_;

public:
	UnaryOp(const MatrixBase<_Child>& child, const _UnaryFunctor& functor = _UnaryFunctor())
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
	CastingOp(const MatrixBase<_Child>& child)
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

CUMAT_NAMESPACE_END

#endif
