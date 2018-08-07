#ifndef __CUMAT_SPARSE_PRODUCT_EVALUATION__
#define __CUMAT_SPARSE_PRODUCT_EVALUATION__

#include "Macros.h"
#include "ProductOp.h"
#include "SparseMatrix.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{

    // CwiseSrcTag * CwiseSrcTag -> SparseDstTag; outer product
    //This handles all dense cwise+matrix inputs and dense matrix output
    //The sparse methods (SparseSrcTag, SparseDstTag) are handled seperately
    template<
        typename _Dst, ProductArgOp _DstOp,
        typename _SrcLeft, ProductArgOp _SrcLeftOp,
        typename _SrcRight, ProductArgOp _SrcRightOp,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<_Dst, SparseDstTag, _DstOp, _SrcLeft, CwiseSrcTag, _SrcLeftOp, _SrcRight, CwiseSrcTag, _SrcRightOp, _AssignmentMode>
    {
        using Op = ProductOp<_SrcLeft, _SrcRight, _SrcLeftOp, _SrcRightOp, _DstOp>;
        using Scalar = typename Op::Scalar;

        static void assign(_Dst& dst, const Op& op) {
            //Check that the input matrices are vectors
            CUMAT_STATIC_ASSERT((Op::TransposedLeft ? Op::RowsLeft : Op::ColumnsLeft == 1),
                "Product evaluation into a sparse matrix is only supported for the outer product of two vectors, left matrix is not a column vector");
            CUMAT_STATIC_ASSERT((Op::TransposedRight ? Op::ColumnsRight : Op::RowsRight == 1),
                "Product evaluation into a sparse matrix is only supported for the outer product of two vectors, right matrix is not a row vector");

            //launch cwise-evaluation
            Assignment<_Dst, Op, _AssignmentMode, typename traits<_Dst>::DstTag, CwiseSrcTag>::assign(dst, op);
        }
    };
    
    namespace
    {
    template <typename L, typename R, typename M, AssignmentMode Mode>
    __global__ void CSRMVKernel(dim3 virtual_size, const L matrix, const R vector, M output)
    {
        typedef typename L::Scalar LeftScalar;
        typedef typename R::Scalar RightScalar;
        typedef typename M::Scalar OutputScalar;
        typedef ProductElementFunctor<LeftScalar, RightScalar, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE> Functor;
        const int* JA = matrix.getOuterIndices().data();
        const int* IA = matrix.getInnerIndices().data();
        const LeftScalar* A = matrix.getData().data();
        CUMAT_KERNEL_1D_LOOP(outer, virtual_size) {
            int start = JA[outer];
            int end = JA[outer + 1];
            if (start>=end) continue;
            OutputScalar value = Functor::mult(A[start], vector.coeff(IA[start], 0, 0, -1));
            for (int i=start+1; i<end; ++i)
            {
                value += Functor::mult(A[i], vector.coeff(IA[i], 0, 0, -1));
            }
            internal::CwiseAssignmentHandler<M, OutputScalar, Mode>::assign(output, value, outer);
        }
    }
    }

    //CwiseSrcTag (Sparse) * CwiseSrcTag (Dense-Vector) -> DenseDstTag (Vector-Vector), sparse matrix-vector product
    //Currently, only non-batched CSR matrices are supported
    //TODO: support also CSC, vector on the left, transposed and conjugated versions
    template<
        typename _Dst,// ProductArgOp _DstOp,
        typename _SrcLeftScalar,// ProductArgOp _SrcLeftOp,
        typename _SrcRight,// ProductArgOp _SrcRightOp,
        AssignmentMode _AssignmentMode
    >
    struct ProductAssignment<
        _Dst, DenseDstTag, ProductArgOp::NONE, 
        SparseMatrix<_SrcLeftScalar, 1, SparseFlags::CSR>, CwiseSrcTag, ProductArgOp::NONE, 
        _SrcRight, CwiseSrcTag, ProductArgOp::NONE, 
        _AssignmentMode>
    {
        using SrcLeft = SparseMatrix<_SrcLeftScalar, 1, SparseFlags::CSR>;
        using Op = ProductOp<SrcLeft, _SrcRight, ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>;
        using Scalar = typename Op::Scalar;

        CUMAT_STATIC_ASSERT((Op::ColumnsRight == 1),
                "SparseMatrix - DenseVector product only supports column vectors as right argument (for now)");
        CUMAT_STATIC_ASSERT((Op::BatchesRight == 1),
                "SparseMatrix - DenseVector does not support batches yet");

        static void assign(_Dst& dst, const Op& op) {
            
            typedef typename _Dst::Type DstActual;
            CUMAT_PROFILING_INC(EvalMatmulSparse);
            CUMAT_PROFILING_INC(EvalAny);
            if (dst.size() == 0) return;
            CUMAT_ASSERT(op.rows() == dst.rows());
            CUMAT_ASSERT(op.cols() == dst.cols());
            CUMAT_ASSERT(op.batches() == dst.batches());

            CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluate SparseMatrix-DenseVector multiplication " << typeid(op.derived()).name();
            CUMAT_LOG(CUMAT_LOG_DEBUG) << " matrix rows=" << op.derived().left().rows() << ", cols=" << op.left().cols();

            //here is now the real logic
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(dst.size());
            CSRMVKernel<SrcLeft, typename _SrcRight::Type, DstActual, _AssignmentMode> 
                <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
                (cfg.virtual_size, op.derived().left().derived(), op.derived().right().derived(), dst.derived());
            CUMAT_CHECK_ERROR();
            CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluation done";
        }
    };
}

CUMAT_NAMESPACE_END

#endif