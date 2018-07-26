#ifndef __CUMAT_SPARSE_EVALUATION__
#define __CUMAT_SPARSE_EVALUATION__

#include "Macros.h"
#include "CwiseOp.h"
#include "SparseMatrix.h"

CUMAT_NAMESPACE_BEGIN

namespace
{
    template <typename T, typename M, AssignmentMode Mode, SparseFlags Flags>
    __global__ void CwiseSparseEvaluationKernel(dim3 virtual_size, const T expr, M matrix)
    {
        const int* JA = matrix.getOuterIndices().data();
        const int* IA = matrix.getInnerIndices().data();
        Index batchStride = matrix.nnz();
        //TODO: Profiling, what is the best way to loop over the batches?
        CUMAT_KERNEL_2D_LOOP(outer, batch, virtual_size) {
            int start = JA[outer];
            int end = JA[outer + 1];
            for (int i=start; i<end; ++i)
            {
                int inner = IA[i];
                Index row = Flags==SparseFlags::CSC ? inner : outer;
                Index col = Flags == SparseFlags::CSC ? outer : inner;
                Index idx = i + batch * batchStride;
                auto val = expr.coeff(row, col, batch, idx);
                internal::CwiseAssignmentHandler<M, decltype(val), Mode>::assign(matrix, val, idx);
            }
        }
    }
}

namespace internal {

    //General assignment for everything that fullfills CwiseSrcTag into SparseDstTag (cwise sparse evaluation)
    //The source expression is only evaluated at the non-zero entries of the target SparseMatrix
    template<typename _Dst, typename _Src, AssignmentMode _Mode>
    struct Assignment<_Dst, _Src, _Mode, SparseDstTag, CwiseSrcTag>
    {
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

            CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluate component wise sparse expression " << typeid(src.derived()).name();
            CUMAT_LOG(CUMAT_LOG_DEBUG) << " rows=" << src.rows() << ", cols=" << src.cols() << ", batches=" << src.batches();

            //here is now the real logic
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig2D(dst.derived().outerSize(), dst.derived().batches());
            CwiseSparseEvaluationKernel<SrcActual, DstActual, _Mode, SparseFlags(DstActual::SparseFlags)> 
                <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> 
                (cfg.virtual_size, src.derived(), dst.derived());
            CUMAT_CHECK_ERROR();
            CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluation done";
        }
    };

}

CUMAT_NAMESPACE_END

#endif