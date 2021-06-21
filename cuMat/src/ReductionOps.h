#ifndef __CUMAT_REDUCTION_OPS_H__
#define __CUMAT_REDUCTION_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Constants.h"
#include "Iterator.h"
#include "ReductionAlgorithmSelection.h"
#include "Errors.h"

#if CUMAT_NVCC==1
#include "../../third-party/cub/cub.cuh"
#endif

#ifndef CUMAT_CUB_DEBUG
#define CUMAT_CUB_DEBUG false
#endif

CUMAT_NAMESPACE_BEGIN

//if CUMAT_UNITTESTS_LAST_REDUCTION is defined,
//the name of the last executed algorithm is stored in the following global variable.
//This global variable has to be declared in some source file as well.
#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
extern std::string LastReductionAlgorithm;
#endif

namespace internal
{
	/**
	 * \brief Generate reduce evaluator / dispatcher.
	 * 
	 * (Partial) specializations must define the function
	 * <code>
	 * static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial);
	 * </code>
	 * 
	 * \tparam _Input input matrix type
	 * \tparam _Output output matrix type
	 * \tparam _Axis the reduction axis, a binary combination of the constants in \ref Axis
	 * \tparam _Op the reduction operator (binary op)
	 * \tparam _Scalar the scalar type
	 * \tparam _Algorithm tag for selecting the algorithm. Valid tags are in the namespace \ref ReductionAlg.
	 */
    template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar, typename _Algorithm>
    struct ReductionEvaluator
    {
        static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial);
    };

	//----------------------------------------------------------------------
	// HELPER STRUCTURE
	// computes the iterators and number of entries and batches
	// given the reduction axis.
	//----------------------------------------------------------------------

	template<typename _Input, typename _Output, int _Axis>
	struct ReductionEvaluatorHelper //creates the iterators and numbers
	{};
	//row reduction
	template<typename _Input, typename _Output>
	struct ReductionEvaluatorHelper<_Input, _Output, Axis::Row>
	{
		static StridedMatrixInputIterator<_Input> iterIn(const MatrixBase<_Input>& in)
		{
			return StridedMatrixInputIterator<_Input>(in, thrust::make_tuple(1, in.rows(), in.rows()*in.cols()));
		}
		static StridedMatrixOutputIterator<_Output> iterOut(_Output& out) 
		{
			return StridedMatrixOutputIterator<_Output>(out, thrust::make_tuple(1, 1, out.cols()));
		}
		static Index numEntries(const MatrixBase<_Input>& in) { return in.rows(); }
		static Index numBatches(const MatrixBase<_Input>& in) { return in.cols()*in.batches(); }
	};
	//column reduction
	template<typename _Input, typename _Output>
	struct ReductionEvaluatorHelper<_Input, _Output, Axis::Column>
	{
		static StridedMatrixInputIterator<_Input> iterIn(const MatrixBase<_Input>& in)
		{
			return StridedMatrixInputIterator<_Input>(in, thrust::make_tuple(in.cols(), 1, in.rows()*in.cols()));
		}
		static StridedMatrixOutputIterator<_Output> iterOut(_Output& out)
		{
			return StridedMatrixOutputIterator<_Output>(out, thrust::make_tuple(1, 1, out.rows()));
		}
		static Index numEntries(const MatrixBase<_Input>& in) { return in.cols(); }
		static Index numBatches(const MatrixBase<_Input>& in) { return in.rows()*in.batches(); }
	};
	//batch reduction
	template<typename _Input, typename _Output>
	struct ReductionEvaluatorHelper<_Input, _Output, Axis::Batch>
	{
		static StridedMatrixInputIterator<_Input> iterIn(const MatrixBase<_Input>& in)
		{
			bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Output>::Flags);
			return StridedMatrixInputIterator<_Input>(in, isRowMajor
				? thrust::make_tuple(in.batches()*in.cols(), in.batches(), 1)
				: thrust::make_tuple(in.batches(), in.batches()*in.rows(), 1));
		}
		static StridedMatrixOutputIterator<_Output> iterOut(_Output& out)
		{
			bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Output>::Flags);
			return StridedMatrixOutputIterator<_Output>(out, isRowMajor
				? thrust::make_tuple(out.cols(), Index(1), 1)
				: thrust::make_tuple(Index(1), out.rows(), 1));
		}
		static Index numEntries(const MatrixBase<_Input>& in) { return in.batches(); }
		static Index numBatches(const MatrixBase<_Input>& in) { return in.rows()*in.cols(); }
	};
	//row+column reduction
	template<typename _Input, typename _Output>
	struct ReductionEvaluatorHelper<_Input, _Output, Axis::Row | Axis::Column>
	{
		static StridedMatrixInputIterator<_Input> iterIn(const MatrixBase<_Input>& in)
		{
			bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Output>::Flags);
			return StridedMatrixInputIterator<_Input>(in, isRowMajor
				? thrust::make_tuple(in.cols(), Index(1), in.cols()*in.rows())
				: thrust::make_tuple(Index(1), in.rows(), in.rows()*in.cols()));
		}
		static typename _Output::Scalar* iterOut(_Output& out)
		{
			return out.data();
		}
		static Index numEntries(const MatrixBase<_Input>& in) { return in.rows()*in.cols(); }
		static Index numBatches(const MatrixBase<_Input>& in) { return in.batches(); }
	};
	//row+batch reduction
	template<typename _Input, typename _Output>
	struct ReductionEvaluatorHelper<_Input, _Output, Axis::Row | Axis::Batch>
	{
		static StridedMatrixInputIterator<_Input> iterIn(const MatrixBase<_Input>& in)
		{
			return StridedMatrixInputIterator<_Input>(in,
				thrust::make_tuple(1, in.rows()*in.batches(), in.rows()));
		}
		static typename _Output::Scalar* iterOut(_Output& out)
		{
			return out.data();
		}
		static Index numEntries(const MatrixBase<_Input>& in) { return in.rows()*in.batches(); }
		static Index numBatches(const MatrixBase<_Input>& in) { return in.cols(); }
	};
	//column+batch reduction
	template<typename _Input, typename _Output>
	struct ReductionEvaluatorHelper<_Input, _Output, Axis::Column | Axis::Batch>
	{
		static StridedMatrixInputIterator<_Input> iterIn(const MatrixBase<_Input>& in)
		{
			return StridedMatrixInputIterator<_Input>(in,
				thrust::make_tuple(in.cols()*in.batches(), 1, in.cols()));
		}
		static typename _Output::Scalar* iterOut(_Output& out)
		{
			return out.data();
		}
		static Index numEntries(const MatrixBase<_Input>& in) { return in.cols()*in.batches(); }
		static Index numBatches(const MatrixBase<_Input>& in) { return in.rows(); }
	};

	//----------------------------------------------------------------------
	// SPECIALIZATIONS for the REDUCTION ALGORITHM
	// (excluding no-op and full reduction. It is handled above)
	//----------------------------------------------------------------------

#if CUMAT_NVCC==1
	//cub::DeviceSegmentedReduce
	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Segmented>
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			int numEntries = static_cast<int>(ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in));
			int numBatches = static_cast<int>(ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in));

			//call cub
			CountingInputIterator<int> iterOffsets(0, numEntries);
			Context& ctx = Context::current();
			size_t temp_storage_bytes = 0;
			CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
				NULL, temp_storage_bytes, 
				iterIn, iterOut, numBatches, iterOffsets, iterOffsets + 1, 
				op, initial, ctx.stream(), CUMAT_CUB_DEBUG));
			DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
			CUMAT_SAFE_CALL(cub::DeviceSegmentedReduce::Reduce(
				static_cast<void*>(temp_storage.pointer()), temp_storage_bytes, 
				iterIn, iterOut, numBatches, iterOffsets, iterOffsets + 1,
				op, initial, ctx.stream(), CUMAT_CUB_DEBUG));

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "Segmented";
#endif
		}
	};

	//cub::DeviceReduce
	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar, int MiniBatch>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Device<MiniBatch> >
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			Index numEntries = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in);
			Index numBatches = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in);

			//TODO: cache substreams in Context
			cudaStream_t substreams[MiniBatch];
			for (int i=0; i<MiniBatch; ++i)
			{
				CUMAT_SAFE_CALL(cudaStreamCreateWithFlags(&substreams[i], cudaStreamNonBlocking));
			}
			Event event;
			size_t temp_storage_bytes = 0;
			DevicePointer<uint8_t> temp_storage[MiniBatch];

			cudaStream_t mainStream = Context::current().stream();
			int Bmin = std::min(int(numBatches), MiniBatch);

			//initialize temporal storage and add sync points
			event.record(mainStream);
			cub::DeviceReduce::Reduce(
				nullptr,
				temp_storage_bytes,
				iterIn,
				iterOut,
				int(numEntries), 
				op, initial,
				substreams[0]);
			for (int b = 0; b < Bmin; ++b)
			{
				event.streamWait(substreams[b]);
				temp_storage[b] = DevicePointer<uint8_t>(temp_storage_bytes);
			}
			//perform reduction
			for (Index b = 0; b < numBatches; ++b)
			{
				const int i = b % MiniBatch;
				cub::DeviceReduce::Reduce(
					temp_storage[i].pointer(),
					temp_storage_bytes,
					iterIn + (numEntries * b),
					iterOut + b,
					int(numEntries), 
					op, initial,
					substreams[i]);
			}
			//add sync points
			for (int b = 0; b < Bmin; ++b)
			{
				event.record(substreams[b]);
				event.streamWait(mainStream);
			}
			//free streams
			for (int i = 0; i < MiniBatch; ++i)
			{
				cudaStreamDestroy(substreams[i]);
			}
			
#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "Device<"+std::to_string(MiniBatch)+">";
#endif
		}
	};

	//cub::DeviceReduce, one stream
	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Device<1> >
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			Index numEntries = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in);
			Index numBatches = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in);

			size_t temp_storage_bytes = 0;
			DevicePointer<uint8_t> temp_storage;
			cudaStream_t mainStream = Context::current().stream();

			//initialize temporal storage
			cub::DeviceReduce::Reduce(
				nullptr,
				temp_storage_bytes,
				iterIn,
				iterOut,
				int(numEntries),
				op, initial,
				mainStream);
			//perform reduction
			temp_storage = DevicePointer<uint8_t>(temp_storage_bytes);
			for (Index b = 0; b < numBatches; ++b)
			{
				cub::DeviceReduce::Reduce(
					temp_storage.pointer(),
					temp_storage_bytes,
					iterIn + (numEntries * b),
					iterOut + b,
					int(numEntries),
					op, initial,
					mainStream);
			}

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "Device<1>";
#endif
		}
	};

	//Thread reduction
	namespace kernels
	{
		template<typename _Input, typename _Output, typename _Op, typename _Scalar>
		__global__ void ReduceThreadKernel(dim3 virtual_size,
			_Input input, _Output output, _Op op, int N)
		{
			CUMAT_KERNEL_1D_LOOP(i, virtual_size)
				const Index O = i * N;
				_Scalar v = input[O];
				for (int n = 1; n < N; ++n) v = op(v, input[O + n]);
				output[i] = v;
			CUMAT_KERNEL_1D_LOOP_END
		}
	}
	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Thread>
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			Index numEntries = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in);
			Index numBatches = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in);

			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(numBatches, 
				kernels::ReduceThreadKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar>);
			kernels::ReduceThreadKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar>
				<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
					cfg.virtual_size, iterIn, iterOut, op, int(numEntries));
			CUMAT_CHECK_ERROR();

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "Thread";
#endif
		}
	};

	//Warp reduction
	namespace kernels
	{
		template<typename _Input, typename _Output, typename _Op, typename _Scalar>
		__global__ void ReduceWarpKernel(dim3 virtual_size,
			_Input input, _Output output, _Op op, _Scalar initial, Index N)
		{
			CUMAT_KERNEL_1D_LOOP(i_, virtual_size)
				const Index i = i_ / 32;//i_ >> 5;
				const int warp = i_ % 32; // i_ & 32;
				const Index O = i * N;
				//local reduce
				_Scalar v = initial;
				for (Index n = warp; n < N; n += 32)
					v = op(v, input[O + n]);
				//final warp reduce
				#pragma unroll
				for (int offset = 16; offset > 0; offset /= 2)
					v = op(v, __shfl_down_sync(0xffffffff, v, offset));
				//write output
				if (warp == 0) output[i] = v;
			CUMAT_KERNEL_1D_LOOP_END
		}
	}
	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Warp>
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			Index numEntries = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in);
			Index numBatches = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in);

			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(numBatches * 32, 
				kernels::ReduceWarpKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar>);
			kernels::ReduceWarpKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar>
				<<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
					cfg.virtual_size, iterIn, iterOut, op, initial, numEntries);
			CUMAT_CHECK_ERROR();

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "Warp";
#endif
		}
	};

	//Block reduction
	namespace kernels
	{
		template<typename _Input, typename _Output, typename _Op, typename _Scalar, int BlockSize>
		__global__ void ReduceBlockKernel(dim3 virtual_size,
			_Input input, _Output output, _Op op, _Scalar initial, Index N)
		{
			const int part = threadIdx.x;

			typedef cub::BlockReduce<_Scalar, BlockSize> BlockReduceT;
			__shared__ typename BlockReduceT::TempStorage temp_storage;

			for (Index i = blockIdx.x; i < virtual_size.x; ++i) {
				const Index O = i * N;
				//local reduce
				_Scalar v = initial;
				for (Index n = part; n < N; n += BlockSize)
					v = op(v, input[O + n]);
				//block reduce
				v = BlockReduceT(temp_storage).Reduce(v, op);
				if (part == 0)
					output[i] = v;
			}
		}
	}
	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar, int BlockSize>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Block<BlockSize> >
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			Index numEntries = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in);
			Index numBatches = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in);

			Context& ctx = Context::current();

			int minGridSize = 0, bestBlockSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, 
				kernels::ReduceBlockKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar, BlockSize>);
			bestBlockSize = BlockSize; //force block size
			minGridSize = std::min(int(numBatches), minGridSize);
			CUMAT_LOG_DEBUG("Best potential occupancy for "
				<< typeid(kernels::ReduceBlockKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar, BlockSize>).name()
				<< " found to be: blocksize=" << bestBlockSize << ", gridSize=" << minGridSize);
			KernelLaunchConfig cfg = {
				dim3(internal::narrow_cast<unsigned>(numBatches), 1, 1),
				dim3(internal::narrow_cast<unsigned>(bestBlockSize), 1, 1),
				dim3(internal::narrow_cast<unsigned>(minGridSize), 1, 1)
			};

			kernels::ReduceBlockKernel<decltype(iterIn), decltype(iterOut), _Op, _Scalar, BlockSize>
				<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (
					cfg.virtual_size, iterIn, iterOut, op, initial, numEntries);
			CUMAT_CHECK_ERROR();

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "Block<" + std::to_string(BlockSize) + ">";
#endif
		}
	};
#endif

	//-------------------------------------------------
	// Automatic algorithm selection
	//-------------------------------------------------

	template<typename _Input, typename _Output, int _Axis, typename _Op, typename _Scalar>
	struct ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Auto>
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			auto iterIn = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterIn(in);
			auto iterOut = ReductionEvaluatorHelper<_Input, _Output, _Axis>::iterOut(out);
			Index numEntries = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numEntries(in);
			Index numBatches = ReductionEvaluatorHelper<_Input, _Output, _Axis>::numBatches(in);

			//dynamically select the algorithm
			ReductionAlgorithm alg;
			if (CUMAT_IS_ROW_MAJOR(internal::traits<_Input>::Flags))
			{
				//row+column is flipped
				if (_Axis == Axis::Column || _Axis == (Axis::Row | Axis::Column))
					alg = ReductionAlgorithmSelection::inner(numBatches, numEntries);
				else if (_Axis == Axis::Batch || _Axis == (Axis::Batch | Axis::Row))
					alg = ReductionAlgorithmSelection::outer(numBatches, numEntries);
				else //column or mixed (row+batch)
					alg = ReductionAlgorithmSelection::middle(numBatches, numEntries);
			} else
			{
				//row+column is normal
				if (_Axis == Axis::Row || _Axis == (Axis::Row | Axis::Column))
					alg = ReductionAlgorithmSelection::inner(numBatches, numEntries);
				else if (_Axis == Axis::Batch || _Axis == (Axis::Batch | Axis::Column))
					alg = ReductionAlgorithmSelection::outer(numBatches, numEntries);
				else //column or mixed (row+batch)
					alg = ReductionAlgorithmSelection::middle(numBatches, numEntries);
			}

			//switch into the actual implementation
			switch(alg)
			{
			case ReductionAlgorithm::Segmented:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Segmented>
					::eval(in, out, op, initial);
				break;
			case ReductionAlgorithm::Thread:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Thread>
					::eval(in, out, op, initial);
				break;
			case ReductionAlgorithm::Warp:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Warp>
					::eval(in, out, op, initial);
				break;
			case ReductionAlgorithm::Block256:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Block<256>>
					::eval(in, out, op, initial);
				break;
			case ReductionAlgorithm::Device1:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Device<1>>
					::eval(in, out, op, initial);
				break;
			case ReductionAlgorithm::Device2:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Device<2>>
					::eval(in, out, op, initial);
				break;
			case ReductionAlgorithm::Device4:
				ReductionEvaluator<_Input, _Output, _Axis, _Op, _Scalar, ReductionAlg::Device<4>>
					::eval(in, out, op, initial);
				break;
			default:
				throw std::runtime_error("unknown dynamic reduction algorithm");
			}
		}
	};

	//----------------------------------------------------------------------
	// SPECIAL CASES
	// (no reduction + full reduction)
	// needs full specialization over the algorithm to avoid the error
	//  "more than one partial specialization matches the template argument list"
	// Sorry, it is quite ugly
	//----------------------------------------------------------------------

	template<typename _Input, typename _Output, int Axis, typename _Op, typename _Scalar>
	struct ReductionEvaluatorSpecial;

	/* No-op reduction, no axis selected -> copy in to out */
	template <typename _Input, typename _Output, typename _Op, typename _Scalar>
	struct ReductionEvaluatorSpecial<_Input, _Output, 0, _Op, _Scalar>
	{
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			internal::Assignment<_Output, _Input, AssignmentMode::ASSIGN,
			                     typename internal::traits<_Output>::DstTag, typename internal::traits<_Input>::SrcTag>
				::assign(out, in);

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "noop";
#endif
		}
	};

	/* full reduction, always use device reduction */
	template <typename _Input, typename _Output, typename _Op, typename _Scalar>
	struct ReductionEvaluatorSpecial<_Input, _Output, Axis::Row | Axis::Column | Axis::Batch, _Op, _Scalar>
	{
#if CUMAT_NVCC==1
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)
		{
			/* create iterators */
			bool isRowMajor = CUMAT_IS_ROW_MAJOR(internal::traits<_Input>::Flags);
			StridedMatrixInputIterator<_Input> iterIn(in, isRowMajor
				                                              ? thrust::make_tuple(
					                                              in.cols(), Index(1), in.cols() * in.rows())
				                                              : thrust::make_tuple(
					                                              Index(1), in.rows(), in.rows() * in.cols()));
			_Scalar* iterOut = out.data();
			int num_items = static_cast<int>(in.rows() * in.cols() * in.batches());
			/* call cub */
			Context& ctx = Context::current();
			size_t temp_storage_bytes = 0;
			CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(
				NULL, temp_storage_bytes,
				iterIn, iterOut, num_items,
				op, initial, ctx.stream()));
			DevicePointer<uint8_t> temp_storage(temp_storage_bytes);
			CUMAT_SAFE_CALL(cub::DeviceReduce::Reduce(
				static_cast<void*>(temp_storage.pointer()), temp_storage_bytes,
				iterIn, iterOut, num_items,
				op, initial, ctx.stream(),
				CUMAT_CUB_DEBUG));

#ifdef CUMAT_UNITTESTS_LAST_REDUCTION
			LastReductionAlgorithm = "full";
#endif
		}
#endif
	};

#define SPECIALIZE_ALG(alg)																						\
																												\
	/* No-op reduction, no axis selected -> copy in to out */													\
	template<typename _Input, typename _Output, typename _Op, typename _Scalar>									\
	struct ReductionEvaluator<_Input, _Output, 0, _Op, _Scalar, alg>											\
	{																											\
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)		\
		{																										\
			ReductionEvaluatorSpecial<_Input, _Output, 0, _Op, _Scalar>::eval(in, out, op, initial);			\
		}																										\
	};																											\
																												\
	/* full reduction, always use device reduction */															\
	template<typename _Input, typename _Output, typename _Op, typename _Scalar>									\
	struct ReductionEvaluator<_Input, _Output, Axis::All, _Op, _Scalar, alg>									\
	{																											\
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)		\
		{																										\
			ReductionEvaluatorSpecial<_Input, _Output, Axis::All, _Op, _Scalar>::eval(in, out, op, initial);	\
		}																										\
	}
#define SPECIALIZE_ALG_PARAM(alg)																				\
																												\
	/* No-op reduction, no axis selected -> copy in to out */													\
	template<typename _Input, typename _Output, typename _Op, typename _Scalar, int N>							\
	struct ReductionEvaluator<_Input, _Output, 0, _Op, _Scalar, alg<N> >										\
	{																											\
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)		\
		{																										\
			ReductionEvaluatorSpecial<_Input, _Output, 0, _Op, _Scalar>::eval(in, out, op, initial);			\
		}																										\
	};																											\
																												\
	/* full reduction, always use device reduction */															\
	template<typename _Input, typename _Output, typename _Op, typename _Scalar, int N>							\
	struct ReductionEvaluator<_Input, _Output, Axis::All, _Op, _Scalar, alg<N> >								\
	{																											\
		static void eval(const MatrixBase<_Input>& in, _Output& out, const _Op& op, const _Scalar& initial)		\
		{																										\
			ReductionEvaluatorSpecial<_Input, _Output, Axis::All, _Op, _Scalar>::eval(in, out, op, initial);	\
		}																										\
	}

	SPECIALIZE_ALG(ReductionAlg::Segmented);
	SPECIALIZE_ALG(ReductionAlg::Thread);
	SPECIALIZE_ALG(ReductionAlg::Warp);
	SPECIALIZE_ALG(ReductionAlg::Auto);
	SPECIALIZE_ALG_PARAM(ReductionAlg::Block);
	SPECIALIZE_ALG_PARAM(ReductionAlg::Device);
	SPECIALIZE_ALG(ReductionAlg::Device<1>);

#undef SPECIALIZE_ALG
#undef SPECIALIZE_ALG_PARAM

	//-------------------------------------------------
	// ASSIGNMENT
	//-------------------------------------------------

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
    template<typename _Child, typename _ReductionOp, typename _Algorithm>
    struct traits<ReductionOp_DynamicSwitched<_Child, _ReductionOp, _Algorithm> >
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
		typedef _Algorithm Algorithm;
    };
}

template<typename _Child, typename _ReductionOp, typename _Algorithm>
class ReductionOp_DynamicSwitched : public MatrixBase<ReductionOp_DynamicSwitched<_Child, _ReductionOp, _Algorithm> >
{
public:
    typedef MatrixBase<ReductionOp_DynamicSwitched<_Child, _ReductionOp, _Algorithm> > Base;
    typedef ReductionOp_DynamicSwitched<_Child, _ReductionOp, _Algorithm> Type;
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
        case 0: internal::ReductionEvaluator<_Child, Derived, 0, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 1: internal::ReductionEvaluator<_Child, Derived, 1, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 2: internal::ReductionEvaluator<_Child, Derived, 2, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 3: internal::ReductionEvaluator<_Child, Derived, 3, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 4: internal::ReductionEvaluator<_Child, Derived, 4, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 5: internal::ReductionEvaluator<_Child, Derived, 5, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 6: internal::ReductionEvaluator<_Child, Derived, 6, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        case 7: internal::ReductionEvaluator<_Child, Derived, 7, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_); break;
        default: throw std::invalid_argument(__FILE__ ":" CUMAT_STR(__LINE__) 
            ": Invalid argument, axis must be between 0 and 7, but is " + std::to_string(axis_));
        }
        CUMAT_LOG_DEBUG("Evaluation done");
    }
};

namespace internal {
    template<typename _Child, typename _ReductionOp, int _Axis, typename _Algorithm>
    struct traits<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis, _Algorithm> >
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
		typedef _Algorithm Algorithm;
    };
}

template<typename _Child, typename _ReductionOp, int _Axis, typename _Algorithm>
class ReductionOp_StaticSwitched : public MatrixBase<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis, _Algorithm> >
{
public:
    typedef MatrixBase<ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis, _Algorithm> > Base;
    typedef ReductionOp_StaticSwitched<_Child, _ReductionOp, _Axis, _Algorithm> Type;
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
        internal::ReductionEvaluator<_Child, Derived, AxisSimplified, _ReductionOp, Scalar, _Algorithm>::eval(child_, m.derived(), op_, initialValue_);
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