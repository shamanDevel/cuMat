#ifndef REDUCTIONS_BENCHMARK
#define REDUCTIONS_BENCHMARK

#include <cuMat/Core>
#include <third-party/cub/device/device_reduce.cuh>
#include <third-party/cub/block/block_reduce.cuh>

CUMAT_NAMESPACE_BEGIN

// I ALWAYS ASSUME COLUMN-MAJOR HERE:
// idx = row + rows * (col + cols * batch)

//########################################################
// BASELINE
//########################################################

/**
 * Default reduce with cuMat. This serves as the baseline
 */
template<typename _Scalar, typename _Op, int _Axis>
void ReduceDefault(
	const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in, 
	Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out, 
	const _Op& op, const _Scalar& initial)
{
	internal::ReductionEvaluator<
		Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>, 
		Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>,
		_Axis, _Op, _Scalar, ReductionAlg::Segmented>
	::eval(in.derived(), out.derived(), op, initial);
}

//########################################################
// HELPER, will become an internal namespace
//########################################################

template<int Axis>
struct BatchedReductionStrides
{
	//Returns the number of entries to reduce along the specified axis
	__host__ __device__ __inline__ static Index NumEntries(Index rows, Index cols, Index batches);
	//Returns the number of batches (non-reduced dimensions)
	__host__ __device__ __inline__ static Index NumBatches(Index rows, Index cols, Index batches);
	//Returns the stride of the entries for the specified axis
	__host__ __device__ __inline__ static Index Stride(Index rows, Index cols, Index batches);
	//Returns the offset into the array for the specified batch (0 <= batch < NumBatches(..) )
	__host__ __device__ __inline__ static Index Offset(Index rows, Index cols, Index batches, Index batch);
};
template<>
struct BatchedReductionStrides<Axis::Batch> //outer index
{
	__host__ __device__ __inline__ static Index NumEntries(Index rows, Index cols, Index batches) { return batches; }
	__host__ __device__ __inline__ static Index NumBatches(Index rows, Index cols, Index batches) { return rows * cols; }
	__host__ __device__ __inline__ static Index Stride(Index rows, Index cols, Index batches) { return rows * cols; }
	__host__ __device__ __inline__ static Index Offset(Index rows, Index cols, Index batches, Index batch) { return batch; }
};
template<>
struct BatchedReductionStrides<Axis::Column> //middle index
{
	__host__ __device__ __inline__ static Index NumEntries(Index rows, Index cols, Index batches) { return cols; }
	__host__ __device__ __inline__ static Index NumBatches(Index rows, Index cols, Index batches) { return rows * batches; }
	__host__ __device__ __inline__ static Index Stride(Index rows, Index cols, Index batches) { return rows; }
	__host__ __device__ __inline__ static Index Offset(Index rows, Index cols, Index batches, Index batch)
	{
		const Index z = batch / rows;
		const Index x = batch % rows;
		return x + rows * cols*z;
	}
};
template<>
struct BatchedReductionStrides<Axis::Row> //inner index
{
	__host__ __device__ __inline__ static Index NumEntries(Index rows, Index cols, Index batches) { return rows; }
	__host__ __device__ __inline__ static Index NumBatches(Index rows, Index cols, Index batches) { return cols * batches; }
	__host__ __device__ __inline__ static Index Stride(Index rows, Index cols, Index batches) { return 1; }
	__host__ __device__ __inline__ static Index Offset(Index rows, Index cols, Index batches, Index batch) { return rows * batch; }
};

/**
 * \brief A strided iterator
 */
template<
	typename InputIteratorT,
	typename OffsetT = ptrdiff_t,
	typename OutputValueT = typename std::iterator_traits<InputIteratorT>::value_type >
class StridedIterator
{
public:
	// Required iterator traits
	typedef StridedIterator self_type; ///< My own type
	typedef OffsetT difference_type; ///< Type to express the result of subtracting one iterator from another
	typedef OutputValueT value_type;
	typedef value_type* pointer;
	typedef value_type& reference;
	using iterator_category = std::random_access_iterator_tag;

protected:
	InputIteratorT child_;
	OffsetT stride_;

public:
	/// Constructor
	__host__ __device__
		StridedIterator(const InputIteratorT& child, OffsetT stride)
		: child_(child)
		, stride_(stride)
	{}

	/// Postfix increment
	__host__ __device__ CUMAT_STRONG_INLINE self_type operator++(int)
	{
		self_type retval = *this;
		child_ += stride_;
		return retval;
	}

	/// Prefix increment
	__host__ __device__ CUMAT_STRONG_INLINE self_type& operator++()
	{
		child_ += stride_;
		return *this;
	}

	/// Indirection
	__device__ CUMAT_STRONG_INLINE value_type operator*() const
	{
		return *child_;
	}

	__device__ CUMAT_STRONG_INLINE reference operator*()
	{
		return *child_;	
	}

	/// Addition
	template <typename Distance>
	__host__ __device__ CUMAT_STRONG_INLINE self_type operator+(Distance n) const
	{
		self_type retval = *this;
		retval.child_ += n*stride_;
		return retval;
	}

	/// Addition assignment
	template <typename Distance>
	__host__ __device__ CUMAT_STRONG_INLINE self_type& operator+=(Distance n)
	{
		child_ += n*stride_;
		return *this;
	}

	/// Subtraction
	template <typename Distance>
	__host__ __device__ CUMAT_STRONG_INLINE self_type operator-(Distance n) const
	{
		self_type retval = *this;
		retval.child_ -= n*stride_;
		return retval;
	}

	/// Subtraction assignment
	template <typename Distance>
	__host__ __device__ CUMAT_STRONG_INLINE self_type& operator-=(Distance n)
	{
		child_ -= n*stride_;
		return *this;
	}

	/// Distance
	__host__ __device__ __forceinline__ difference_type operator-(self_type other) const
	{
		return (child_ - other.child_) / stride_;
	}

	/// Array subscript
	template <typename Distance>
	__device__ __forceinline__ value_type operator[](Distance n) const
	{
		return child_[n*stride_];
	}

	/// Equal to
	__host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
	{
		return (child_ == rhs.child_);
	}

	/// Not equal to
	__host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
	{
		return (child_ != rhs.child_);
	}
};
template<typename InputIteratorT, typename OffsetT>
StridedIterator<InputIteratorT, OffsetT>
make_strided_iterator(const InputIteratorT& child, OffsetT stride)
{
	return StridedIterator<InputIteratorT, OffsetT>(child, stride);
}

//########################################################
// NEW ALGORITHMS
//########################################################

/**
 * Batched reduction performed by launching multiple device-wide reductions in separate streams.
 * \tparam MiniBatch specifies the number of streams in parallel
 */
template<typename _Op, typename _Scalar, int _Axis, int MiniBatch = 16>
struct ReduceDevice
{
	static void run(
		const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in,
		Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out,
		const _Op& op, const _Scalar& initial)
	{
		const Index N = BatchedReductionStrides<_Axis>::NumEntries(in.rows(), in.cols(), in.batches());
		const Index B = BatchedReductionStrides<_Axis>::NumBatches(in.rows(), in.cols(), in.batches());
		const Index S = BatchedReductionStrides<_Axis>::Stride(in.rows(), in.cols(), in.batches());
		const _Scalar* input = in.data();
		_Scalar* output = out.data();

		//TODO: cache substreams in Context
		Context substreams[MiniBatch];
		Event event;
		size_t temp_storage_bytes = 0;
		DevicePointer<uint8_t> temp_storage[MiniBatch];

		cudaStream_t mainStream = Context::current().stream();
		int Bmin = std::min(int(B), MiniBatch);

		//initialize temporal storage and add sync points
		event.record(mainStream);
		cub::DeviceReduce::Reduce(
			nullptr,
			temp_storage_bytes,
			make_strided_iterator(input, S),
			output,
			int(N), op, initial,
			substreams[0].stream());
		for (int b=0; b<Bmin; ++b)
		{
			event.streamWait(substreams[b].stream());
			temp_storage[b] = DevicePointer<uint8_t>(temp_storage_bytes, substreams[b]);
		}
		//perform reduction
		for (Index b=0; b<B; ++b)
		{
			const int i = b % MiniBatch;
			Index O = BatchedReductionStrides<_Axis>::Offset(in.rows(), in.cols(), in.batches(), b);
			cub::DeviceReduce::Reduce(
				temp_storage[i].pointer(),
				temp_storage_bytes,
				make_strided_iterator(input + O, S),
				output + b,
				int(N), op, initial,
				substreams[i].stream());
		}
		//add sync points
		for (int b = 0; b < Bmin; ++b)
		{
			event.record(substreams[b].stream());
			event.streamWait(mainStream);
		}
	}
};
template<typename _Op, typename _Scalar, int _Axis>
struct ReduceDevice<_Op, _Scalar, _Axis, 1>
{
	static void run(
		const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in,
		Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out,
		const _Op& op, const _Scalar& initial)
	{
		const Index N = BatchedReductionStrides<_Axis>::NumEntries(in.rows(), in.cols(), in.batches());
		const Index B = BatchedReductionStrides<_Axis>::NumBatches(in.rows(), in.cols(), in.batches());
		const Index S = BatchedReductionStrides<_Axis>::Stride(in.rows(), in.cols(), in.batches());
		const _Scalar* input = in.data();
		_Scalar* output = out.data();

		size_t temp_storage_bytes = 0;
		DevicePointer<uint8_t> temp_storage;
		cudaStream_t mainStream = Context::current().stream();

		//initialize temporal storage
		cub::DeviceReduce::Reduce(
			nullptr,
			temp_storage_bytes,
			make_strided_iterator(input, S),
			output,
			int(N), op, initial,
			mainStream);
		temp_storage = DevicePointer<uint8_t>(temp_storage_bytes);
		//perform reduction
		for (Index b = 0; b < B; ++b)
		{
			Index O = BatchedReductionStrides<_Axis>::Offset(in.rows(), in.cols(), in.batches(), b);
			cub::DeviceReduce::Reduce(
				temp_storage.pointer(),
				temp_storage_bytes,
				make_strided_iterator(input + O, S),
				output + b,
				int(N), op, initial,
				mainStream);
		}
	}
};

namespace kernels
{
	template<typename _Input, typename _Output, typename _Op, typename _Scalar, int _Axis>
	__global__ void ReduceThreadKernel(dim3 virtual_size,
		_Input input, _Output output, _Op op,
		int N, Index S, Index rows, Index cols, Index batches)
	{
		CUMAT_KERNEL_1D_LOOP(i, virtual_size)
			const Index O = BatchedReductionStrides<_Axis>::Offset(rows, cols, batches, i);
			_Scalar v = input[O];
			for (int n = 1; n < N; ++n) v = op(v, input[n*S + O]);
			output[i] = v;
		CUMAT_KERNEL_1D_LOOP_END
	}
}
/**
 * Batched reduction - thread wise.
 * Each thread reduces one data row.
 */
template<typename _Op, typename _Scalar, int _Axis>
void ReduceThread(
	const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in,
	Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out,
	const _Op& op, const _Scalar& initial)
{
	const Index N = BatchedReductionStrides<_Axis>::NumEntries(in.rows(), in.cols(), in.batches());
	const Index B = BatchedReductionStrides<_Axis>::NumBatches(in.rows(), in.cols(), in.batches());
	const Index S = BatchedReductionStrides<_Axis>::Stride(in.rows(), in.cols(), in.batches());
	const _Scalar* input = in.data();
	_Scalar* output = out.data();

	Context& ctx = Context::current();
	KernelLaunchConfig cfg = ctx.createLaunchConfig1D(B, kernels::ReduceThreadKernel<const _Scalar*, _Scalar*, _Op, _Scalar, _Axis>);
	kernels::ReduceThreadKernel<const _Scalar*, _Scalar*, _Op, _Scalar, _Axis> 
		<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
			cfg.virtual_size, input, output, op,
			int(N), S, in.rows(), in.cols(), in.batches());
	CUMAT_CHECK_ERROR();
}


namespace kernels
{
	template<typename _Input, typename _Output, typename _Op, typename _Scalar, int _Axis>
	__global__ void ReduceThreadWarp(dim3 virtual_size,
		_Input input, _Output output, _Op op, _Scalar initial,
		int N, Index S, Index rows, Index cols, Index batches)
	{
		CUMAT_KERNEL_1D_LOOP(i_, virtual_size)
			const Index i = i_ / 32;//i_ >> 5;
			const int warp = i_ % 32; // i_ & 32;
			const Index O = BatchedReductionStrides<_Axis>::Offset(rows, cols, batches, i);
			//printf("i=%5d, w=%2d, O=%5d, N=%d, S=%d\n", int(i), warp, int(O), int(N), int(S));
			//local reduce
			_Scalar v = initial;
			for (int n = warp; n < N; n += 32)
				v = op(v, input[n*S + O]);
			//final warp reduce
			#pragma unroll
			for (int offset = 16; offset > 0; offset /= 2)
				v += __shfl_down_sync(0xffffffff, v, offset);
			//write output
			if (warp == 0) output[i] = v;
		CUMAT_KERNEL_1D_LOOP_END
	}
}
/**
 * Batched reduction - warp wise.
 * Each thread reduces one data row.
 */
template<typename _Op, typename _Scalar, int _Axis>
void ReduceWarp(
	const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in,
	Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out,
	const _Op& op, const _Scalar& initial)
{
	const Index N = BatchedReductionStrides<_Axis>::NumEntries(in.rows(), in.cols(), in.batches());
	const Index B = BatchedReductionStrides<_Axis>::NumBatches(in.rows(), in.cols(), in.batches());
	const Index S = BatchedReductionStrides<_Axis>::Stride(in.rows(), in.cols(), in.batches());
	const _Scalar* input = in.data();
	_Scalar* output = out.data();

	Context& ctx = Context::current();
	KernelLaunchConfig cfg = ctx.createLaunchConfig1D(B*32, kernels::ReduceThreadWarp<const _Scalar*, _Scalar*, _Op, _Scalar, _Axis>);
	kernels::ReduceThreadWarp<const _Scalar*, _Scalar*, _Op, _Scalar, _Axis>
		<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
			cfg.virtual_size, input, output, op, initial,
			int(N), S, in.rows(), in.cols(), in.batches());
	CUMAT_CHECK_ERROR();
}


namespace kernels
{
	template<typename _Input, typename _Output, typename _Op, typename _Scalar, int _Axis, int BlockSize>
	__global__ void ReduceBlockKernel(dim3 virtual_size,
		_Input input, _Output output, _Op op, _Scalar initial,
		int N, Index S, Index rows, Index cols, Index batches)
	{
		const int part = threadIdx.x;

		typedef cub::BlockReduce<_Scalar, BlockSize> BlockReduceT;
		__shared__ typename BlockReduceT::TempStorage temp_storage;

		for (Index i = blockIdx.x; i < virtual_size.x; ++i) {
			const Index O = BatchedReductionStrides<_Axis>::Offset(rows, cols, batches, i);
			//local reduce
			_Scalar v = initial;
			for (int n = part; n < N; n += BlockSize)
				v = op(v, input[n*S + O]);
			//block reduce
			v = BlockReduceT(temp_storage).Reduce(v, op);
			if (part == 0)
				output[i] = v;
		}
	}
}
/**
 * Batched reduction - block wise.
 * Each thread reduces one data row.
 */
template<typename _Op, typename _Scalar, int _Axis, int BlockSize>
void ReduceBlock(
	const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in,
	Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out,
	const _Op& op, const _Scalar& initial)
{
	const Index N = BatchedReductionStrides<_Axis>::NumEntries(in.rows(), in.cols(), in.batches());
	const Index B = BatchedReductionStrides<_Axis>::NumBatches(in.rows(), in.cols(), in.batches());
	const Index S = BatchedReductionStrides<_Axis>::Stride(in.rows(), in.cols(), in.batches());
	const _Scalar* input = in.data();
	_Scalar* output = out.data();

	Context& ctx = Context::current();

	int minGridSize = 0, bestBlockSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, kernels::ReduceBlockKernel<const _Scalar*, _Scalar*, _Op, _Scalar, _Axis, BlockSize>);
	bestBlockSize = BlockSize; //force block size
	minGridSize = std::min(int(B), minGridSize);
	CUMAT_LOG_DEBUG("Best potential occupancy for " 
		<< typeid(kernels::ReduceBlockKernel<_Scalar*, _Scalar*, _Op, _Scalar, _Axis, BlockSize>).name() 
		<< " found to be: blocksize=" << bestBlockSize << ", gridSize=" << minGridSize);
	KernelLaunchConfig cfg = {
		dim3(B, 1, 1),
		dim3(bestBlockSize, 1, 1),
		dim3(minGridSize, 1, 1)
	};

	kernels::ReduceBlockKernel<const _Scalar*, _Scalar*, _Op, _Scalar, _Axis, BlockSize>
		<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
			cfg.virtual_size, input, output, op, initial,
			int(N), S, in.rows(), in.cols(), in.batches());
	CUMAT_CHECK_ERROR();
}


CUMAT_NAMESPACE_END

#endif