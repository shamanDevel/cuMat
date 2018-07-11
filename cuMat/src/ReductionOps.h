#ifndef __CUMAT_REDUCTION_OPS_H__
#define __CUMAT_REDUCTION_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Constants.h"
#include "Iterator.h"

#include <cub/cub.cuh>
#include <stdexcept>

#ifndef CUMAT_CUB_DEBUG
#define CUMAT_CUB_DEBUG false
#endif

CUMAT_NAMESPACE_BEGIN

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
            in.template evalTo<_Output, AssignmentMode::ASSIGN>(out);
        }
    };

    //row reduction
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            StridedMatrixInputIterator<_Input> iterIn(in, thrust::make_tuple(1, in.rows(), in.rows()*in.cols()));
            StridedMatrixOutputIterator<_Output> iterOut(out, thrust::make_tuple(1, 1, in.cols()));
            CountingInputIterator<int> iterOffsets(0, static_cast<int>(in.rows()));
            int num_segments = static_cast<int>(in.cols() * in.batches());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets+1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };

    //column reduction
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Column, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            StridedMatrixInputIterator<_Input> iterIn(in, thrust::make_tuple(in.cols(), 1, in.rows()*in.cols()));
            StridedMatrixOutputIterator<_Output> iterOut(out, thrust::make_tuple(1, 1, in.rows()));
            CountingInputIterator<int> iterOffsets(0, static_cast<int>(in.cols()));
            int num_segments = static_cast<int>(in.rows() * in.batches());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };

    //reduction over batches
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Output>::Flags);
            StridedMatrixInputIterator<_Input> iterIn(in, isRowMajor
                ? thrust::make_tuple(in.batches()*in.cols(), in.batches(), 1)
                : thrust::make_tuple(in.batches(), in.batches()*in.rows(), 1));
            StridedMatrixOutputIterator<_Output> iterOut(out, isRowMajor
                ? thrust::make_tuple(in.cols(), Index(1), 1)
                : thrust::make_tuple(Index(1), in.rows(), 1));
            CountingInputIterator<int> iterOffsets(0, static_cast<int>(in.batches()));
            int num_segments = static_cast<int>(in.rows() * in.cols());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };

    //reduction over rows and columns
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row | ReductionAxis::Column, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Input>::Flags);
            StridedMatrixInputIterator<_Input> iterIn(in, isRowMajor
                ? thrust::make_tuple(in.cols(), Index(1), in.cols()*in.rows())
                : thrust::make_tuple(Index(1), in.rows(), in.rows()*in.cols()));
            _Scalar* iterOut = out.data();
            CountingInputIterator<int> iterOffsets(0, static_cast<int>(in.rows()*in.cols()));
            int num_segments = static_cast<int>(in.batches());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };

    //reduction over rows and batches
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Row | ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            StridedMatrixInputIterator<_Input> iterIn(in, thrust::make_tuple(1, in.rows()*in.batches(), in.rows()));
            _Scalar* iterOut = out.data();
            CountingInputIterator<int> iterOffsets(0, static_cast<int>(in.rows() * in.batches()));
            int num_segments = static_cast<int>(in.cols());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };

    //reduction over colums and batches
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, ReductionAxis::Column | ReductionAxis::Batch, _Op, _Scalar>
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
        {
            //create iterators
            StridedMatrixInputIterator<_Input> iterIn(in, thrust::make_tuple(in.cols()*in.batches(), 1, in.cols()));
            _Scalar* iterOut = out.data();
            CountingInputIterator<int> iterOffsets(0, static_cast<int>(in.cols() * in.batches()));
            int num_segments = static_cast<int>(in.rows());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_segments, iterOffsets, iterOffsets + 1, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
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
            StridedMatrixInputIterator<_Input> iterIn(in, isRowMajor 
                ? thrust::make_tuple(in.cols(), Index(1), in.cols()*in.rows())
                : thrust::make_tuple(Index(1), in.rows(), in.rows()*in.cols()));
            _Scalar* iterOut = out.data();
            int num_items = static_cast<int>(in.rows() * in.cols() * in.batches());
            //call cub
            Context& ctx = Context::current();
            size_t temp_storage_bytes = 0;
            CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, iterIn, iterOut, num_items, op, initial, ctx.stream()));
            DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
            CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, iterIn, iterOut, num_items, op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
        }
    };
} //end internal

// now come the real ops

namespace internal {
    template<typename _Child, typename _ReductionOp>
    struct traits<ReductionOp_DynamicSwitched<_Child, _ReductionOp> >
    {
        using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            RowsAtCompileTime = Dynamic,
            ColsAtCompileTime = Dynamic,
            BatchesAtCompileTime = Dynamic,
            AccessFlags = 0 //must be completely evaluated
        };
    };
}

template<typename _Child, typename _ReductionOp>
class ReductionOp_DynamicSwitched : public MatrixBase<ReductionOp_DynamicSwitched<_Child, _ReductionOp> >
{
public:
    typedef MatrixBase<ReductionOp_DynamicSwitched<_Child, _ReductionOp> > Base;
    using Scalar = typename internal::traits<_Child>::Scalar;
    enum
    {
        Flags = internal::traits<_Child>::Flags,
        Rows = Dynamic,
        Columns = Dynamic,
        Batches = Dynamic
    };

    using Base::size;
    using Base::derived;
    using Base::eval_t;

protected:
    const _Child child_;
    const int axis_;
    const _ReductionOp op_;
    const Scalar initialValue_;

public:
    ReductionOp_DynamicSwitched(const MatrixBase<_Child>& child, int axis, const _ReductionOp& op, const Scalar& initialValue)
        : child_(child.derived())
        , axis_(axis)
        , op_(op)
        , initialValue_(initialValue)
    {}

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const
    {
        return (axis_ & ReductionAxis::Row) ? 1 : child_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        return (axis_ & ReductionAxis::Column) ? 1 : child_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return (axis_ & ReductionAxis::Batch) ? 1 : child_.batches();
    }

    template<typename Derived, AssignmentMode Mode>
    void evalTo(MatrixBase<Derived>& m) const
    {
        //TODO: Handle different assignment modes
        static_assert(Mode == AssignmentMode::ASSIGN, "Currently, only AssignmentMode::ASSIGN is supported");

        CUMAT_PROFILING_INC(EvalReduction);
        CUMAT_PROFILING_INC(EvalAny);
        if (size() == 0) return;
        CUMAT_ASSERT(rows() == m.rows());
        CUMAT_ASSERT(cols() == m.cols());
        CUMAT_ASSERT(batches() == m.batches());

        CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluate reduction expression " << typeid(derived()).name();
        CUMAT_LOG(CUMAT_LOG_DEBUG) << " rows=" << m.rows() << ", cols=" << m.cols() << ", batches=" << m.batches()
            << ", axis=" << ((axis_ & ReductionAxis::Row) ? "R" : "") << ((axis_ & ReductionAxis::Column) ? "C" : "") << ((axis_ & ReductionAxis::Batch) ? "B" : "");

        //runtime switch to the implementations
        switch (axis_)
        {
        case 0: internal::ReductionEvaluator<_Child, Derived, 0, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 1: internal::ReductionEvaluator<_Child, Derived, 1, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 2: internal::ReductionEvaluator<_Child, Derived, 2, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 3: internal::ReductionEvaluator<_Child, Derived, 3, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 4: internal::ReductionEvaluator<_Child, Derived, 4, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 5: internal::ReductionEvaluator<_Child, Derived, 5, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 6: internal::ReductionEvaluator<_Child, Derived, 6, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        case 7: internal::ReductionEvaluator<_Child, Derived, 7, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_); break;
        default: throw std::invalid_argument(__FILE__ ":" CUMAT_STR(__LINE__) 
            ": Invalid argument, axis must be between 0 and 7, but is " + std::to_string(axis_));
        }
        CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluation done";
    }
};

namespace internal {
    template<typename _Child, typename _ReductionOp, int _Axis>
    struct traits<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis> >
    {
        using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            RowsAtCompileTime = ((_Axis & ReductionAxis::Row) ? 1 : internal::traits<_Child>::RowsAtCompileTime),
            ColsAtCompileTime = ((_Axis & ReductionAxis::Column) ? 1 : internal::traits<_Child>::ColsAtCompileTime),
            BatchesAtCompileTime = ((_Axis & ReductionAxis::Batch) ? 1 : internal::traits<_Child>::BatchesAtCompileTime),
            AccessFlags = 0
        };
    };
}

template<typename _Child, typename _ReductionOp, int _Axis>
class ReductionOp_StaticSwitched : public MatrixBase<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis> >
{
public:
    typedef MatrixBase<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis> > Base;
    using Scalar = typename internal::traits<_Child>::Scalar;
    enum
    {
        Flags = internal::traits<_Child>::Flags,
        Rows = Dynamic,
        Columns = Dynamic,
        Batches = Dynamic
    };

    using Base::size;
    using Base::derived;
    using Base::eval_t;

protected:
    const _Child child_;
    const _ReductionOp op_;
    const Scalar initialValue_;

public:
    ReductionOp_StaticSwitched(const MatrixBase<_Child>& child, const _ReductionOp& op, const Scalar& initialValue)
        : child_(child.derived())
        , op_(op)
        , initialValue_(initialValue)
    {
        CUMAT_STATIC_ASSERT(_Axis >= 0 && _Axis <= 7, "Axis must be between 0 and 7");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const
    {
        return (_Axis & ReductionAxis::Row) ? 1 : child_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        return (_Axis & ReductionAxis::Column) ? 1 : child_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return (_Axis & ReductionAxis::Batch) ? 1 : child_.batches();
    }

    template<typename Derived, AssignmentMode Mode>
    void evalTo(MatrixBase<Derived>& m) const
    {
        //TODO: Handle different assignment modes
        static_assert(Mode == AssignmentMode::ASSIGN, "Currently, only AssignmentMode::ASSIGN is supported");

        CUMAT_PROFILING_INC(EvalReduction);
        CUMAT_PROFILING_INC(EvalAny);
        if (size() == 0) return;
        CUMAT_ASSERT(rows() == m.rows());
        CUMAT_ASSERT(cols() == m.cols());
        CUMAT_ASSERT(batches() == m.batches());

        CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluate reduction expression " << typeid(derived()).name();
        CUMAT_LOG(CUMAT_LOG_DEBUG) << " rows=" << m.rows() << ", cols=" << m.cols() << ", batches=" << m.batches()
            << ", axis=" << ((_Axis & ReductionAxis::Row) ? "R" : "") << ((_Axis & ReductionAxis::Column) ? "C" : "") << ((_Axis & ReductionAxis::Batch) ? "B" : "");

        //compile-time switch to the implementations
        internal::ReductionEvaluator<_Child, Derived, _Axis, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_);
        CUMAT_LOG(CUMAT_LOG_DEBUG) << "Evaluation done";
    }
};

namespace functor
{
    // REDUCTION FUNCTORS
    // first three copied from CUB

    /**
    * \brief Default sum functor
    */
    template <typename T>
    struct Sum
    {
        /// Boolean sum operator, returns <tt>a + b</tt>
        __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
        {
            return a + b;
        }
    };

    /**
    * \brief Default max functor
    */
    template <typename T>
    struct Max
    {
        /// Boolean max operator, returns <tt>(a > b) ? a : b</tt>
        __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
        {
            return CUB_MAX(a, b);
        }
    };


    /**
    * \brief Default min functor
    */
    template <typename T>
    struct Min
    {
        /// Boolean min operator, returns <tt>(a < b) ? a : b</tt>
        __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
        {
            return CUB_MIN(a, b);
        }
    };

    /**
    * \brief Default product functor
    */
    template <typename T>
    struct Prod
    {
        /// Boolean product operator, returns <tt>a * b</tt>
        __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
        {
            return a * b;
        }
    };

    /**
    * \brief Default logical AND functor, only works on booleans
    */
    template <typename T>
    struct LogicalAnd
    {
        /// Boolean AND operator, returns <tt>a && b</tt>
        __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) const
        {
            return a && b;
        }
    };

    /**
    * \brief Default logical OR functor, only works on booleans
    */
    template <typename T>
    struct LogicalOr
    {
        /// Boolean OR operator, returns <tt>a || b</tt>
        __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) const
        {
            return a || b;
        }
    };

    /**
    * \brief Default bitwise AND functor, only works on integers
    */
    template <typename T>
    struct BitwiseAnd
    {
        /// bitwise AND operator, returns <tt>a & b</tt>
        __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
        {
            return a & b;
        }
    };

    /**
    * \brief Default bitwise AND functor, only works on integers
    */
    template <typename T>
    struct BitwiseOr
    {
        /// bitwise OR operator, returns <tt>a | b</tt>
        __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
        {
            return a | b;
        }
    };
}

CUMAT_NAMESPACE_END

#endif
