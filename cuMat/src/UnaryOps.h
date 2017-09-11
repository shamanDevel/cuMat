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
		using Scalar = internal::traits<_Child>::Scalar;
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

	__device__ CUMAT_STRONG_INLINE const Scalar& coeff(Index row, Index col, Index batch) const
	{
		return functor_(child_.derived().coeff(row, col, batch), row, col, batch);
	}
};

namespace functor
{
#define DECLARE_FUNCTOR(Name) \
	template<typename _Scalar> class UnaryMathFunctor_ ## Name

#define DEFINE_GENERAL_FUNCTOR(Name, Fn) \
	template<typename _Scalar> \
	class UnaryMathFunctor_ ## Name \
	{ \
	public: \
		__device__ CUMAT_STRONG_INLINE const _Scalar& operator()(const _Scalar& x, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

#define DEFINE_FUNCTOR(Name, Scalar, Fn) \
	template<> \
	class UnaryMathFunctor_ ## Name <Scalar> \
	{ \
	public: \
		__device__ CUMAT_STRONG_INLINE const _Scalar& operator()(const _Scalar& x, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

#define DEFINE_FUNCTOR_FLOAT(Name, Fn) \
	DEFINE_FUNCTOR(Name, float, Fn);   \
	DEFINE_FUNCTOR(Name, double, Fn);

	//DECLARE_FUNCTOR(negate); //this is already done in ForwardDeclarations.h

	DEFINE_GENERAL_FUNCTOR(cwiseNegate, (-x));
	DEFINE_GENERAL_FUNCTOR(cwiseAbs, abs(x));
	DEFINE_GENERAL_FUNCTOR(cwiseInverse, 1/x);

#undef DECLARE_FUNCTOR
#undef DEFINE_GENERAL_FUNCTOR
#undef DEFINE_FUNCTOR
#undef DEFINE_FUNCTOR_FLOAT
}

CUMAT_NAMESPACE_END

#endif
