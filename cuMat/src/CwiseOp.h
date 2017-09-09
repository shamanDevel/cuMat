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
		//By using a 1D-loop over the linear index,
		//the target matrix can determine the order of rows, columns and batches.
		//E.g. by storage order (row major / column major)
		//Later, this may come in hand if sparse matrices or diagonal matrices are allowed
		//that only evaluate certain elements.
		CUMAT_KERNEL_1D_LOOP(index, virtual_size) {
			Index i, j, k;
			matrix.index(index, i, j, k);
			//printf("eval at row=%d, col=%d, batch=%d, index=%d\n", (int)i, (int)j, (int)k, (int)matrix.index(i, j, k));
			matrix.rawCoeff(index) = expr.coeff(i, j, k);
		}
	}
}

/**
 * \brief Base class of all component-wise expressions.
 * It defines the evaluation logic.
 * 
 * A component-wise expression can be evaluated to any object that
 *  - inherits MatrixBase
 *  - defines a <code>__host__ Index size() const</code> method that returns the number of entries
 *  - defines a <code>__device__ void index(Index index, Index& row, Index& col, Index& batch) const</code>
 *    method to convert from raw index (from 0 to size()-1) to row, column and batch index
 *  - defines a <code>__Device__ Scalar& rawCoeff(Index index)</code> method
 *    that is used to write the results back.
 * 
 * Currently, the following classes support this interface and can therefore be used
 * as the left-hand-side of a component-wise expression:
 *  - Matrix
 *  - MatrixBlock
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
		KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.size());
		EvaluationKernel<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream()>>>(cfg.virtual_size, derived(), m.derived());
		CUMAT_CHECK_ERROR();
		CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluation done";
	}
};

CUMAT_NAMESPACE_END

#endif