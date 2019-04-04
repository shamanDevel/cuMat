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
	//TODO: use the various reduction algorithms from the benchmark.
	//Switch between the algorithms based on the timings in the benchmarks.

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
            internal::Assignment<_Output, _Input, AssignmentMode::ASSIGN, 
                typename internal::traits<_Output>::DstTag, typename internal::traits<_Input>::SrcTag>
            ::assign(out, in);
        }
    };

    //row reduction
    template<typename _Input, typename _Output, typename _Op, typename _Scalar>
    struct ReductionEvaluator<_Input, _Output, Axis::Row, _Op, _Scalar>
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
    struct ReductionEvaluator<_Input, _Output, Axis::Column, _Op, _Scalar>
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
    struct ReductionEvaluator<_Input, _Output, Axis::Batch, _Op, _Scalar>
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
    struct ReductionEvaluator<_Input, _Output, Axis::Row | Axis::Column, _Op, _Scalar>
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
    struct ReductionEvaluator<_Input, _Output, Axis::Row | Axis::Batch, _Op, _Scalar>
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
    struct ReductionEvaluator<_Input, _Output, Axis::Column | Axis::Batch, _Op, _Scalar>
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
    struct ReductionEvaluator<_Input, _Output, Axis::Row | Axis::Column | Axis::Batch, _Op, _Scalar>
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

    //We can already declare the assignment operator here
    struct ReductionSrcTag {};
    template<typename _Dst, typename _Src>
    struct Assignment<_Dst, _Src, AssignmentMode::ASSIGN, DenseDstTag, ReductionSrcTag>
    {
        static void assign(_Dst& dst, const _Src& src)
        {
            //for now, use the simple version and delegate to evalTo.
            src.template evalTo<typename _Dst::Type, AssignmentMode::ASSIGN>(dst.derived());
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
        typedef ReductionSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}

template<typename _Child, typename _ReductionOp>
class ReductionOp_DynamicSwitched : public MatrixBase<ReductionOp_DynamicSwitched<_Child, _ReductionOp> >
{
public:
    typedef MatrixBase<ReductionOp_DynamicSwitched<_Child, _ReductionOp> > Base;
    typedef ReductionOp_DynamicSwitched<_Child, _ReductionOp> Type;
    CUMAT_PUBLIC_API
    using Base::size;

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
        return (axis_ & Axis::Row) ? 1 : child_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        return (axis_ & Axis::Column) ? 1 : child_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return (axis_ & Axis::Batch) ? 1 : child_.batches();
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

        CUMAT_LOG_DEBUG("Evaluate reduction expression " << typeid(derived()).name()
			<< "\n rows=" << m.rows() << ", cols=" << m.cols() << ", batches=" << m.batches()
            << ", axis=" << ((axis_ & Axis::Row) ? "R" : "") << ((axis_ & Axis::Column) ? "C" : "") << ((axis_ & Axis::Batch) ? "B" : ""));

		//simplify axis, less segments are better
		const int axisSimplified =
			axis_ == 0 ? 0 : (
				axis_ | (internal::traits<_Child>::RowsAtCompileTime==1 ? Axis::Row : 0)
				      | (internal::traits<_Child>::ColsAtCompileTime==1 ? Axis::Column : 0)
				      | (internal::traits<_Child>::BatchesAtCompileTime==1 ? Axis::Batch : 0)
			);

        //runtime switch to the implementations
        switch (axisSimplified)
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
        CUMAT_LOG_DEBUG("Evaluation done");
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
            RowsAtCompileTime = ((_Axis & Axis::Row) ? 1 : internal::traits<_Child>::RowsAtCompileTime),
            ColsAtCompileTime = ((_Axis & Axis::Column) ? 1 : internal::traits<_Child>::ColsAtCompileTime),
            BatchesAtCompileTime = ((_Axis & Axis::Batch) ? 1 : internal::traits<_Child>::BatchesAtCompileTime),
            AccessFlags = 0
        };
        typedef ReductionSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}

template<typename _Child, typename _ReductionOp, int _Axis>
class ReductionOp_StaticSwitched : public MatrixBase<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis> >
{
public:
    typedef MatrixBase<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis> > Base;
    typedef ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis> Type;
    CUMAT_PUBLIC_API
    using Base::size;

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
        return (_Axis & Axis::Row) ? 1 : child_.rows();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const
    {
        return (_Axis & Axis::Column) ? 1 : child_.cols();
    }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const
    {
        return (_Axis & Axis::Batch) ? 1 : child_.batches();
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

		//simplify axis, less segments are better
		constexpr int AxisSimplified =
			_Axis == 0 ? 0 : (
				_Axis | (internal::traits<_Child>::RowsAtCompileTime==1 ? Axis::Row : 0)
				      | (internal::traits<_Child>::ColsAtCompileTime==1 ? Axis::Column : 0)
				      | (internal::traits<_Child>::BatchesAtCompileTime==1 ? Axis::Batch : 0)
			);

		CUMAT_LOG_DEBUG("Evaluate reduction expression " << typeid(derived()).name()
			<< "\n rows=" << m.rows() << "(" << internal::traits<_Child>::RowsAtCompileTime << ")"
    		<< ", cols=" << m.cols() << "(" << internal::traits<_Child>::ColsAtCompileTime  << ")" 
    		<< ", batches=" << m.batches() << "(" << internal::traits<_Child>::BatchesAtCompileTime << ")"
            << ", axis=" << ((AxisSimplified & Axis::Row) ? "R" : "") << ((AxisSimplified & Axis::Column) ? "C" : "") << ((AxisSimplified & Axis::Batch) ? "B" : ""));

        //compile-time switch to the implementations
        internal::ReductionEvaluator<_Child, Derived, AxisSimplified, _ReductionOp, Scalar>::eval(child_, m.derived(), op_, initialValue_);
        CUMAT_LOG_DEBUG("Evaluation done");
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
