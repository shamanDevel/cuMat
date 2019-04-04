template<typename _Input, typename _Output, typename _Op, typename _Scalar, int _Axis, int BlockSize>
__global__ void ReduceBlockKernel(
	_Input input, _Output output, _Op op,
	Index I, int N, Index S)
{
	const int part = threadIdx.x;

	typedef cub::BlockReduce<_Scalar, BlockSize> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;

	for (Index i = blockIdx.x; i < I; ++i) {
		const Index O = Offset(rows, cols, batches, i);
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
//Here, the block size is fixed to BlockSize, also passed as template parameters.
//Values that were experminented with are 128, 256, 512, 1024. 1024 is the maximal number of threads per SM.
//The grid size is determined by cudaOccupancyMaxPotentialBlockSize.