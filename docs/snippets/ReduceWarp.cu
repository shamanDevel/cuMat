template<typename _Input, typename _Output, typename _Op, typename _Scalar, int _Axis>
__global__ void ReduceWarpKernel(
	_Input input, _Output output, _Op op,
	Index I_, int N, Index S)
{
	CUMAT_KERNEL_1D_LOOP(i_, I_)
		const Index i = i_ / 32;
		const int warp = i_ % 32;
		const Index O = Offset(i);
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
//Warp size is always 32 in all Nvidia cards.
//Launched with optimal threads-per-block and grid size for I_=I*32 threads, 
//as determined by cudaOccupancyMaxPotentialBlockSize