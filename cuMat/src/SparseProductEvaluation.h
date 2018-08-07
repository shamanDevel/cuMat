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
    
}

CUMAT_NAMESPACE_END

#endif