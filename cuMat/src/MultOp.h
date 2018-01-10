#ifndef __CUMAT_MULT_OP_H__
#define __CUMAT_MULT_OP_H__

#include <type_traits>
#include <cuda.h>

#include "Macros.h"
#include "Constants.h"
#include "ForwardDeclarations.h"
#include "MatrixBase.h"
#include "Logging.h"
#include "CublasApi.h"

CUMAT_NAMESPACE_BEGIN


template<typename _Left, typename _Right, bool _TransposedLeft, bool _TransposedRight, bool _TransposedOutput>
class MultOp : public MatrixBase<MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput>>
{
public:
    typedef MatrixBase<MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput>> Base;
    using Base::Scalar;
    using Type = MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, _TransposedOutput>;

    enum
    {
        FlagsLeft = internal::traits<_Left>::Flags,
        RowsLeft = internal::traits<_Left>::RowsAtCompileTime,
        ColumnsLeft = internal::traits<_Left>::ColsAtCompileTime,
        BatchesLeft = internal::traits<_Left>::BatchesAtCompileTime,

        FlagsRight = internal::traits<_Right>::Flags,
        RowsRight = internal::traits<_Right>::RowsAtCompileTime,
        ColumnsRight = internal::traits<_Right>::ColsAtCompileTime,
        BatchesRight = internal::traits<_Right>::BatchesAtCompileTime,

        Flags = Flags::ColumnMajor, //TODO: pick best flag
        Rows = RowsLeft,
        Columns = ColumnsRight,
        Batches = BatchesLeft, //TODO: add broadcasting of batches
    };

    using Base::size;
    using Base::derived;
    using Base::eval_t;
    using Base::rows;
    using Base::cols;
    using Base::batches;

private:
    //wrapper that evaluate any cwise-expression to a matrix.
    //For BLAS we need the actual evaluated matrices.
    typedef typename MatrixReadWrapper<_Left, AccessFlags::ReadDirect>::type left_wrapped_t;
    typedef typename MatrixReadWrapper<_Right, AccessFlags::ReadDirect>::type right_wrapped_t;
    const left_wrapped_t left_;
    const right_wrapped_t right_;

public:
    MultOp(const MatrixBase<_Left>& left, const MatrixBase<_Right>& right)
        : left_(left.derived()), right_(right.derived())
    {
        CUMAT_STATIC_ASSERT((std::is_same<typename internal::traits<_Left>::Scalar, typename internal::traits<_Right>::Scalar>::value),
            "No implicit casting is allowed in binary operations.");

        if (ColumnsLeft == Dynamic || RowsRight == Dynamic)
        {
            CUMAT_ASSERT_ARGUMENT(left.cols() == right.rows());
        }
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES((ColumnsLeft >= 1 && RowsRight >= 1), ColumnsLeft == RowsRight), "matrix sizes not compatible");

        if (BatchesLeft == Dynamic || BatchesRight == Dynamic) {
            CUMAT_ASSERT_ARGUMENT(left.batches() == right.batches());
        }
        CUMAT_STATIC_ASSERT(!(BatchesLeft > 1 && BatchesRight > 1) || (BatchesLeft == BatchesRight), "matrix sizes don't match");
    }

    template<int _Rows, int _Columns, int _Batches, int _Flags>
    void evalImpl(Matrix<Scalar, _Rows, _Columns, _Batches, CUctx_flags_enum>& mat) const
    {
        CUMAT_ASSERT_ARGUMENT(mat.rows() == rows());
        CUMAT_ASSERT_ARGUMENT(mat.cols() == cols());
        CUMAT_ASSERT_ARGUMENT(mat.batches() == batches());

        //call cuBLAS
        //TODO
    }

    template<typename Derived>
    void evalTo(MatrixBase<Derived>& m) const
    {
        evalImpl(m.derived());
    }

    //Overwrites transpose()
    typedef MultOp<_Left, _Right, _TransposedLeft, _TransposedRight, !_TransposedOutput> transposed_mult_t;
    const transposed_mult_t& transpose() const
    {
        //transposition just changes the _TransposedOutput-flag
        return transposed_mult_t(left_, right_);
    }
};


CUMAT_NAMESPACE_END


#endif