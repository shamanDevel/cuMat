#include "reductions.h"

#include <cuMat/src/Macros.h>
#include <cuMat/src/Errors.h>
#include "reductions.cuh"

using namespace cuMat;

template<typename _Scalar, int _Axis>
Timings benchmark(Index rows, Index cols, Index batches, bool compare, bool optimize)
{
	//timing events
	Event start, end;
	cudaStream_t stream = Context::current().stream();
	Timings timings = {};

	//create source matrix
	typedef Matrix<_Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor> Mat;
	Mat source(rows, cols, batches);
	SimpleRandom rnd;
	rnd.fillUniform(source, _Scalar(0), _Scalar(50));

	//create output matrix
	Mat target1(_Axis & Axis::Row ? 1 : rows, _Axis & Axis::Column ? 1 : cols, _Axis & Axis::Batch ? 1 : batches);
	Mat target2(_Axis & Axis::Row ? 1 : rows, _Axis & Axis::Column ? 1 : cols, _Axis & Axis::Batch ? 1 : batches);

	//baseline
	target1.setZero();
	start.record(stream);
	target1.inplace() = source.template sum<_Axis>();
	end.record(stream);
	timings.baseline = Event::elapsedTime(start, end);

	//thread
	target2.setZero();
	start.record(stream);
	ReduceThread<functor::Sum<_Scalar>, _Scalar, _Axis>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceThread produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.thread = Event::elapsedTime(start, end);

	//warp
	target2.setZero();
	start.record(stream);
	ReduceWarp<functor::Sum<_Scalar>, _Scalar, _Axis>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceWarp produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.warp = Event::elapsedTime(start, end);

	//block - 64
	target2.setZero();
	start.record(stream);
	ReduceBlock<functor::Sum<_Scalar>, _Scalar, _Axis, 64>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceBlock-64 produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.block64 = Event::elapsedTime(start, end);

	//block - 128
	target2.setZero();
	start.record(stream);
	ReduceBlock<functor::Sum<_Scalar>, _Scalar, _Axis, 128>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceBlock-128 produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.block128 = Event::elapsedTime(start, end);

	//block - 256
	target2.setZero();
	start.record(stream);
	ReduceBlock<functor::Sum<_Scalar>, _Scalar, _Axis, 256>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceBlock-256 produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.block256 = Event::elapsedTime(start, end);

	//block - 512
	target2.setZero();
	start.record(stream);
	ReduceBlock<functor::Sum<_Scalar>, _Scalar, _Axis, 512>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceBlock-512 produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.block512 = Event::elapsedTime(start, end);

	//block - 1024
	target2.setZero();
	start.record(stream);
	ReduceBlock<functor::Sum<_Scalar>, _Scalar, _Axis, 1024>(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
	end.record(stream);
	if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
		std::cerr << "ReduceBlock-1024 produced a wrong result!" << std::endl;
		std::cerr << "Expected: " << target1 << std::endl;
		std::cerr << "Actual: " << target2 << std::endl;
	}
	timings.block1024 = Event::elapsedTime(start, end);

	int numBatches = _Axis == Axis::Row ? cols*batches 
		: _Axis == Axis::Column ? rows*batches
		: rows*cols;
	if (numBatches > 256 && optimize) {
		//with more than 256 batches, ReduceDevice just becomes crazily slow.
		//Set the timings to infinity, otherwise the benchmarks will never finish.
		timings.device1 = 1e20;
		timings.device2 = 1e20;
		timings.device4 = 1e20;
		timings.device8 = 1e20;
		timings.device16 = 1e20;
		timings.device32 = 1e20;
	} else {
		//device-1
		target2.setZero();
		start.record(stream);
		ReduceDevice<functor::Sum<_Scalar>, _Scalar, _Axis, 1>::run(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
		end.record(stream);
		if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
			std::cerr << "ReduceDevice-8 produced a wrong result!" << std::endl;
			std::cerr << "Expected: " << target1 << std::endl;
			std::cerr << "Actual: " << target2 << std::endl;
		}
		timings.device1 = Event::elapsedTime(start, end);

		//device-2
		target2.setZero();
		start.record(stream);
		ReduceDevice<functor::Sum<_Scalar>, _Scalar, _Axis, 2>::run(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
		end.record(stream);
		if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
			std::cerr << "ReduceDevice-8 produced a wrong result!" << std::endl;
			std::cerr << "Expected: " << target1 << std::endl;
			std::cerr << "Actual: " << target2 << std::endl;
		}
		timings.device2 = Event::elapsedTime(start, end);

		//device-4
		target2.setZero();
		start.record(stream);
		ReduceDevice<functor::Sum<_Scalar>, _Scalar, _Axis, 4>::run(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
		end.record(stream);
		if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
			std::cerr << "ReduceDevice-8 produced a wrong result!" << std::endl;
			std::cerr << "Expected: " << target1 << std::endl;
			std::cerr << "Actual: " << target2 << std::endl;
		}
		timings.device4 = Event::elapsedTime(start, end);

		//device-8
		target2.setZero();
		start.record(stream);
		ReduceDevice<functor::Sum<_Scalar>, _Scalar, _Axis, 8>::run(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
		end.record(stream);
		if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
			std::cerr << "ReduceDevice-8 produced a wrong result!" << std::endl;
			std::cerr << "Expected: " << target1 << std::endl;
			std::cerr << "Actual: " << target2 << std::endl;
		}
		timings.device8 = Event::elapsedTime(start, end);

		//device-16
		target2.setZero();
		start.record(stream);
		ReduceDevice<functor::Sum<_Scalar>, _Scalar, _Axis, 16>::run(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
		end.record(stream);
		if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
			std::cerr << "ReduceDevice-16 produced a wrong result!" << std::endl;
			std::cerr << "Expected: " << target1 << std::endl;
			std::cerr << "Actual: " << target2 << std::endl;
		}
		timings.device16 = Event::elapsedTime(start, end);

		//device-32
		target2.setZero();
		start.record(stream);
		ReduceDevice<functor::Sum<_Scalar>, _Scalar, _Axis, 32>::run(source, target2, functor::Sum<_Scalar>(), _Scalar(0));
		end.record(stream);
		if (compare && !static_cast<bool>(((target1 - target2) <= _Scalar(1e-5)).all())) {
			std::cerr << "ReduceDevice-32 produced a wrong result!" << std::endl;
			std::cerr << "Expected: " << target1 << std::endl;
			std::cerr << "Actual: " << target2 << std::endl;
		}
		timings.device32 = Event::elapsedTime(start, end);
	}

	return timings;
}

template<typename _Scalar, int _Axis>
void benchmarkAndPrint(Index rows, Index cols, Index batches, bool compare)
{
	Timings t = benchmark<_Scalar, _Axis>(rows, cols, batches, compare);
	std::cout << "Type=" << typeid(_Scalar).name()
		<< ", Axis=" << (_Axis == Axis::Row ? "Row" : (_Axis == Axis::Column ? "Column" : "Batch"))
		<< ", Dimensions=(" << rows << "," << cols << "," << batches << ") ->"
		<< t << std::endl;
}

/*
 * Launch benchmarks with template-parameters passed as strings.
 * scalar: "int", "float"
 * axis: "row", "col", "batch"
 */
Timings benchmark(Index rows, Index cols, Index batches,
	const std::string& scalar, const std::string& axis, bool compare, bool optimize)
{
	if (scalar == "int")
	{
		if (axis == "row")
			return benchmark<int, Axis::Row>(rows, cols, batches, compare, optimize);
		if (axis == "col")
			return benchmark<int, Axis::Column>(rows, cols, batches, compare, optimize);
		if (axis == "batch")
			return benchmark<int, Axis::Batch>(rows, cols, batches, compare, optimize);
		throw std::runtime_error(("Unsupported axis: " + axis).c_str());
	}
	if (scalar == "float")
	{
		if (axis == "row")
			return benchmark<float, Axis::Row>(rows, cols, batches, compare, optimize);
		if (axis == "col")
			return benchmark<float, Axis::Column>(rows, cols, batches, compare, optimize);
		if (axis == "batch")
			return benchmark<float, Axis::Batch>(rows, cols, batches, compare, optimize);
		throw std::runtime_error(("Unsupported axis: " + axis).c_str());
	}
	throw std::runtime_error(("Unsupported scalar datatype: " + scalar).c_str());
}