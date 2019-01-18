#ifndef __CUMAT_SPARSE_EXPRESSION_OP__
#define __CUMAT_SPARSE_EXPRESSION_OP__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "MatrixBase.h"
#include "SparseMatrixBase.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{
    template<typename _Child, int _SparseFlags>
    struct traits<SparseExpressionOp<_Child, _SparseFlags> >
    {
        using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            SFlags = _SparseFlags,
            Flags = internal::traits<_Child>::Flags,
            RowsAtCompileTime = internal::traits<_Child>::RowsAtCompileTime,
            ColsAtCompileTime = internal::traits<_Child>::ColsAtCompileTime,
            BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise
        };
        typedef CwiseSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}

template<typename _Child, int _SparseFlags>
class SparseExpressionOp : public SparseMatrixBase<SparseExpressionOp<_Child, _SparseFlags>>
{
public:
    typedef SparseExpressionOp<_Child, _SparseFlags> Type;
	typedef SparseMatrixBase<Type> Base;
    CUMAT_PUBLIC_API
    enum
    {
        SFlags = _SparseFlags
    };

    using typename Base::StorageIndex;
    using Base::rows;
    using Base::cols;
    using Base::batches;
    using Base::nnz;
    using Base::size;

protected:
    typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
	const child_wrapped_t child_;

public:
    SparseExpressionOp(const MatrixBase<_Child>& child, const SparsityPattern<_SparseFlags>& sparsityPattern)
        : Base(sparsityPattern, child.batches())
        , child_(child.derived())
	{
        CUMAT_ASSERT_DIMENSION(child.rows() == sparsityPattern.rows);
        CUMAT_ASSERT_DIMENSION(child.cols() == sparsityPattern.cols);
    }

	__device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
	{
		return child_.derived().coeff(row, col, batch, index), row, col, batch;
	}

    __device__ CUMAT_STRONG_INLINE Scalar getSparseCoeff(Index row, Index col, Index batch, Index index) const
    {
        return child_.derived().coeff(row, col, batch, index);
    }
};

CUMAT_NAMESPACE_END


#endif