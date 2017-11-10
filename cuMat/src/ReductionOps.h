#ifndef __CUMAT_REDUCTION_OPS_H__
#define __CUMAT_REDUCTION_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Iterator.h"

#include <cub/cub.cuh>

#ifndef CUMAT_CUB_DEBUG
#define CUMAT_CUB_DEBUG false
#endif

CUMAT_NAMESPACE_BEGIN

enum ReductionAxis
{
    Row = 1,
    Column = 2,
    Batch = 4
};

namespace internal
{

    template<typename _Input, typename _Output, int axis, typename _Op, typename _Scalar>
    struct ReductionEvaluator
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial);
    };

    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, 0, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //No-op reduction, no axis selected -> copy in to out
            in.evalTo(out);
        }
    };

    //row reduction
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Input>::Flags);
            StridedMatrixIterator<_Input> iterIn(in, thrust::make_tuple(1, in.rows(), in.rows()*in.cols()));
            StridedMatrixIterator<_Input> iterOut(out, thrust::make_tuple(1, 1, in.cols()));
            CountingInputIterator<int> iterOffsets(0, in.rows());
            int num_segments = in.cols() * in.batches();
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets+1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };

    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Column, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {

        }
    };

    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {

        }
    };

    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row | ReductionAxis::Column, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {

        }
    };

    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row | ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {

        }
    };

    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Column | ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {

        }
    };

    //full reduction
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Input>::Flags);
            StridedMatrixIterator<_Input> iterIn(in, isRowMajor 
                ? thrust::make_tuple(in.cols(), 1, in.cols()*in.rows())
                : thrust::make_tuple(1, in.rows(), in.rows()*in.cols()));
            _Scalar* iterOut = out.data();
            int num_items = in.rows() * in.cols() * in.batches();
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_items, op, initial, ctx.stream()));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_items, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };
}

CUMAT_NAMESPACE_END

#endif
