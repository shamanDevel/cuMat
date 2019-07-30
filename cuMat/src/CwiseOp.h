#ifndef __CUMAT_CWISE_OP_H__
#define __CUMAT_CWISE_OP_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "MatrixBase.h"
#include "Context.h"
#include "Logging.h"
#include "Profiling.h"

CUMAT_NAMESPACE_BEGIN

namespace internal {
    //compound assignment
    template <typename M, typename V, AssignmentMode Mode>
    struct CwiseAssignmentHandler
    {
        static __device__ CUMAT_STRONG_INLINE void assign(M& matrix, V value, Index index)
        {
            matrix.setRawCoeff(index, value);
        }
    };

#define CUMAT_DECLARE_ASSIGNMENT_HANDLER(mode, op1, op2)                    \
    template <typename M, typename V>                                       \
    struct CwiseAssignmentHandler<M, V, AssignmentMode:: mode >             \
    {                                                                       \
        static __device__ CUMAT_STRONG_INLINE void assign(M& matrix, V value, Index index, std::integral_constant<int, AccessFlags::RWCwiseRef>) /*reference access*/       \
        {                                                                   \
            matrix.rawCoeff(index) op1 value;                               \
        }                                                                   \
        static __device__ CUMAT_STRONG_INLINE void assign(M& matrix, V value, Index index, std::integral_constant<int, AccessFlags::RWCwise>) /*read-write access*/         \
        {                                                                    \
            matrix.setRawCoeff(index) = matrix.getRawCoeff(index) op2 value;  \
        }                                                                    \
        static __device__ CUMAT_STRONG_INLINE void assign(M& matrix, V value, Index index, std::integral_constant<int, AccessFlags::RWCwiseRef | AccessFlags::RWCwise>) /*both is possible, use reference*/ \
        {                                                                   \
            matrix.rawCoeff(index) op1 value;                               \
        }                                                                   \
        static __device__ CUMAT_STRONG_INLINE void assign(M& matrix, V value, Index index)  \
        {                                                                   \
            assign(matrix, value, index, std::integral_constant<int, traits<M>::AccessFlags & (AccessFlags::RWCwiseRef | AccessFlags::RWCwise)>());     \
        }                                                                   \
    };

    CUMAT_DECLARE_ASSIGNMENT_HANDLER(ADD, +=, +)
    CUMAT_DECLARE_ASSIGNMENT_HANDLER(SUB, -=, -)
    CUMAT_DECLARE_ASSIGNMENT_HANDLER(MUL, *=, *)
    CUMAT_DECLARE_ASSIGNMENT_HANDLER(DIV, /=, /)
    CUMAT_DECLARE_ASSIGNMENT_HANDLER(MOD, %=, %)
    CUMAT_DECLARE_ASSIGNMENT_HANDLER(AND, &=, &)
    CUMAT_DECLARE_ASSIGNMENT_HANDLER(OR, |=, |)

    //partial specialization for normal assignment
    template <typename M, typename V>
    struct CwiseAssignmentHandler<M, V, AssignmentMode::ASSIGN>
    {
        static __device__ CUMAT_STRONG_INLINE void assign(M& matrix, V value, Index index)
        {
            matrix.setRawCoeff(index, value);
        }
    };

	namespace kernels
	{
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void CwiseEvaluationKernel(dim3 virtual_size, const T expr, M matrix)
		{
			//By using a 1D-loop over the linear index,
			//the target matrix can determine the order of rows, columns and batches.
			//E.g. by storage order (row major / column major)
			//Later, this may come in hand if sparse matrices or diagonal matrices are allowed
			//that only evaluate certain elements.
			CUMAT_KERNEL_1D_LOOP(index, virtual_size)

				Index i, j, k;
				matrix.index(index, i, j, k);

				//there seems to be a bug in CUDA if the result of expr.coeff is directly passed to setRawCoeff.
				//By saving it in a local variable, this is prevented
				auto val = expr.coeff(i, j, k, index);
				internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, index);

			CUMAT_KERNEL_1D_LOOP_END
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
 *  - defines a <code>__Device__ void setRawCoeff(Index index, const Scalar& newValue)</code> method
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
    typedef _Derived Type;
	typedef MatrixBase<_Derived> Base;
    CUMAT_PUBLIC_API
	using Base::rows;
	using Base::cols;
	using Base::batches;
	using Base::size;

	__device__ CUMAT_STRONG_INLINE const Scalar& coeff(Index row, Index col, Index batch, Index index) const
	{
		return derived().coeff(row, col, batch, index);
	}

};

namespace internal
{
    //General assignment for everything that fullfills CwiseSrcTag into DenseDstTag (cwise dense evaluation)
    template<typename _Dst, typename _Src, AssignmentMode _Mode>
    struct Assignment<_Dst, _Src, _Mode, DenseDstTag, CwiseSrcTag>
    {
        static void assign(_Dst& dst, const _Src& src)
        {
#if CUMAT_NVCC==1
            typedef typename _Dst::Type DstActual;
            typedef typename _Src::Type SrcActual;
            CUMAT_PROFILING_INC(EvalCwise);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(src.rows() == dst.rows());
            CUMAT_ASSERT(src.cols() == dst.cols());
            CUMAT_ASSERT(src.batches() == dst.batches());

            CUMAT_LOG_DEBUG("Evaluate component wise expression " << typeid(src.derived()).name()
				<< "\n rows=" << src.rows() << ", cols=" << src.cols() << ", batches=" << src.batches());

            //here is now the real logic
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(static_cast<unsigned int>(dst.size()), kernels::CwiseEvaluationKernel<SrcActual, DstActual, _Mode>);
            kernels::CwiseEvaluationKernel<SrcActual, DstActual, _Mode> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (cfg.virtual_size, src.derived(), dst.derived());
            CUMAT_CHECK_ERROR();
            CUMAT_LOG_DEBUG("Evaluation done");
#else
			CUMAT_ERROR_IF_NO_NVCC(general_component_wise_evaluation)
#endif
        }
    };
}

CUMAT_NAMESPACE_END

#endif
