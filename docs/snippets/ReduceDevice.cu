template<typename _Op, typename _Scalar, int _Axis, int MiniBatch = 16>
void ReduceDevice(
	const Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& in,
	Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor>& out,
	const _Op& op, const _Scalar& initial)
{
	const Index N = NumEntries(in.rows(), in.cols(), in.batches());
	const Index B = NumBatches(in.rows(), in.cols(), in.batches());
	const Index S = Stride(in.rows(), in.cols(), in.batches());
	const _Scalar* input = in.data();
	_Scalar* output = out.data();

	//create substreams and event for synchronization
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
		Index O = Offset(in.rows(), in.cols(), in.batches(), b);
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