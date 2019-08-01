#ifndef __CUMAT_SPARSE_EVALUATION__
#define __CUMAT_SPARSE_EVALUATION__

#include "Macros.h"
#include "CwiseOp.h"
#include "SparseMatrix.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{
	namespace kernels
	{
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void CwiseCSREvaluationKernel(dim3 virtual_size, const T expr, M matrix)
		{
			const int* JA = matrix.getSparsityPattern().JA.data();
			const int* IA = matrix.getSparsityPattern().IA.data();
			Index batchStride = matrix.getSparsityPattern().nnz;
			//TODO: Profiling, what is the best way to loop over the batches?
			CUMAT_KERNEL_2D_LOOP(outer, batch, virtual_size)
				int start = JA[outer];
				int end = JA[outer + 1];
				for (int i = start; i < end; ++i)
				{
					int inner = IA[i];
					Index row = outer;
					Index col = inner;
					Index idx = i + batch * batchStride;
					auto val = expr.coeff(row, col, batch, idx);
					internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
				}
			CUMAT_KERNEL_2D_LOOP_END
		}
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void CwiseCSCEvaluationKernel(dim3 virtual_size, const T expr, M matrix)
		{
			const int* JA = matrix.getSparsityPattern().JA.data();
			const int* IA = matrix.getSparsityPattern().IA.data();
			Index batchStride = matrix.getSparsityPattern().nnz;
			//TODO: Profiling, what is the best way to loop over the batches?
			CUMAT_KERNEL_2D_LOOP(outer, batch, virtual_size)
				int start = JA[outer];
				int end = JA[outer + 1];
				for (int i = start; i < end; ++i)
				{
					int inner = IA[i];
					Index row = inner;
					Index col = outer;
					Index idx = i + batch * batchStride;
					auto val = expr.coeff(row, col, batch, idx);
					internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
				}
			CUMAT_KERNEL_2D_LOOP_END
		}
		template <typename T, typename M, AssignmentMode Mode>
		__global__ void CwiseELLPACKEvaluationKernel(dim3 virtual_size, const T expr, M matrix)
		{
			const SparsityPattern<SparseFlags::ELLPACK>::IndexMatrix indices = matrix.getSparsityPattern().indices;
			Index nnzPerRow = matrix.getSparsityPattern().nnzPerRow;
			Index rows = matrix.getSparsityPattern().rows;
			Index batchStride = rows * nnzPerRow;
			//TODO: Profiling, what is the best way to loop over the batches?
			CUMAT_KERNEL_2D_LOOP(row, batch, virtual_size)
				for (int i = 0; i < nnzPerRow; ++i)
				{
					Index col = indices.coeff(row, i, 0, -1);
					if (col < 0) continue; //TODO: test if it is faster to continue reading (and set col=0) and discard before assignment
					Index idx = row + i * rows + batch * batchStride;
					auto val = expr.coeff(row, col, batch, idx);
					internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
				}
			CUMAT_KERNEL_2D_LOOP_END
		}
	}
}

namespace internal {

#if CUMAT_NVCC==1
    //General assignment for everything that fulfills CwiseSrcTag into SparseDstTag (cwise sparse evaluation)
    //The source expression is only evaluated at the non-zero entries of the target SparseMatrix
    template<typename _Dst, typename _Src, AssignmentMode _Mode>
    struct Assignment<_Dst, _Src, _Mode, SparseDstTag, CwiseSrcTag>
    {
    private:
		static void assign(_Dst& dst, const _Src& src, std::integral_constant<int, SparseFlags::CSR>)
		{
			//here is now the real logic
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig2D(static_cast<unsigned int>(dst.derived().outerSize()), static_cast<unsigned int>(dst.derived().batches()),
				kernels::CwiseCSREvaluationKernel<typename _Src::Type, typename _Dst::Type, _Mode>);
			kernels::CwiseCSREvaluationKernel<typename _Src::Type, typename _Dst::Type, _Mode>
				<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
				(cfg.virtual_size, src.derived(), dst.derived());
			CUMAT_CHECK_ERROR();
		}
		static void assign(_Dst& dst, const _Src& src, std::integral_constant<int, SparseFlags::CSC>)
		{
			//here is now the real logic
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig2D(static_cast<unsigned int>(dst.derived().outerSize()), static_cast<unsigned int>(dst.derived().batches()),
				kernels::CwiseCSCEvaluationKernel<typename _Src::Type, typename _Dst::Type, _Mode>);
			kernels::CwiseCSCEvaluationKernel<typename _Src::Type, typename _Dst::Type, _Mode>
				<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
				(cfg.virtual_size, src.derived(), dst.derived());
			CUMAT_CHECK_ERROR();
		}
		static void assign(_Dst& dst, const _Src& src, std::integral_constant<int, SparseFlags::ELLPACK>)
		{
			//here is now the real logic
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig2D(static_cast<unsigned int>(dst.derived().outerSize()), static_cast<unsigned int>(dst.derived().batches()),
				kernels::CwiseELLPACKEvaluationKernel<typename _Src::Type, typename _Dst::Type, _Mode>);
			kernels::CwiseELLPACKEvaluationKernel<typename _Src::Type, typename _Dst::Type, _Mode>
				<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
				(cfg.virtual_size, src.derived(), dst.derived());
			CUMAT_CHECK_ERROR();
		}

    public:
        static void assign(_Dst& dst, const _Src& src)
        {
            typedef typename _Dst::Type DstActual;
            typedef typename _Src::Type SrcActual;
            CUMAT_PROFILING_INC(EvalCwiseSparse);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(src.rows() == dst.rows());
            CUMAT_ASSERT(src.cols() == dst.cols());
            CUMAT_ASSERT(src.batches() == dst.batches());

            CUMAT_LOG_DEBUG("Evaluate component wise sparse expression " << typeid(src.derived()).name()
				<< "\n rows=" << src.rows() << ", cols=" << src.cols() << ", batches=" << src.batches());
			assign(dst, src, std::integral_constant<int, DstActual::SFlags>());
            CUMAT_LOG_DEBUG("Evaluation done");
        }
    };
#endif
}

CUMAT_NAMESPACE_END

#endif