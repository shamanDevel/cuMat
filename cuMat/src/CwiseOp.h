#ifndef __CUMAT_CWISE_OP_H__
#define __CUMAT_CWISE_OP_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "MatrixBase.h"
#include "Context.h"
#include "Logging.h"

CUMAT_NAMESPACE_BEGIN

namespace
{
	template <typename T, typename M>
	__global__ void EvaluationKernel(dim3 virtual_size, const T expr, M matrix)
	{
		CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size) {
			//printf("eval at row=%d, col=%d, batch=%d, index=%d\n", (int)i, (int)j, (int)k, (int)matrix.index(i, j, k));
			matrix.coeff(i, j, k) = expr.coeff(i, j, k);
		}
	}
}

/**
 * \brief Base class of all component-wise expressions.
 * It defines the evaluation logic.
 * 
 * \tparam _Derived the type of the derived expression
 */
template<typename _Derived>
class CwiseOp : public MatrixBase<_Derived>
{
public:
	typedef MatrixBase<_Derived> Base;
	using Base::Scalar;
	using Base::rows;
	using Base::cols;
	using Base::batches;
	using Base::size;
	using Base::derived;
	using Base::eval_t;

	__device__ CUMAT_STRONG_INLINE const Scalar& coeff(Index row, Index col, Index batch) const
	{
		return derived().coeff(row, col, batch);
	}

	template<typename Derived>
	void evalTo(MatrixBase<Derived>& m) const
	{
		if (size() == 0) return;
		CUMAT_ASSERT(rows() == m.rows());
		CUMAT_ASSERT(cols() == m.cols());
		CUMAT_ASSERT(batches() == m.batches());

		CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluate component wise expression " << typeid(derived()).name();
		CUMAT_LOG(CUMAT_LOG_DEBUG) << " rows=" << m.rows() << ", cols=" << m.cols() << ", batches=" << m.batches();

		//here is now the real logic
		Context& ctx = Context::current();
		KernelLaunchConfig cfg = ctx.createLaunchConfig3D(m.rows(), m.cols(), m.batches());
		EvaluationKernel<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream()>>>(cfg.virtual_size, derived(), m.derived());
		CUMAT_CHECK_ERROR();
		CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluation done";
	}
};

CUMAT_NAMESPACE_END

#endif