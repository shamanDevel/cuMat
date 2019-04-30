template<typename _Input, typename _Output, typename _Op, typename _Scalar, int _Axis>
__global__ void ReduceThreadKernel(
	_Input input, _Output output, _Op op,
	Index I, int N, Index S)
{
	CUMAT_KERNEL_1D_LOOP(i, I)
		const Index O = Offset(i);
		_Scalar v = input[O];
		for (int n = 1; n < N; ++n) v = op(v, input[n*S + O]);
		output[i] = v;
	CUMAT_KERNEL_1D_LOOP_END
}
//Launched with optimal threads-per-block and grid size for I threads, 
//as determined by cudaOccupancyMaxPotentialBlockSize