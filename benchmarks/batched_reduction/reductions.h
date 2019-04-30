#ifndef __REDUCTIONS_HOST__
#define __REDUCTIONS_HOST__

#include <ostream>
#include <string>
#include <exception>
#include <string.h>

#include <cuMat/src/ForwardDeclarations.h>

//timings in milliseconds
struct Timings
{
	float baseline;
	float thread;
	float warp;
	float block64;
	float block128;
	float block256;
	float block512;
	float block1024;
	float device1;
	float device2;
	float device4;
	float device8;
	float device16;
	float device32;

	void reset()
	{
		memset(this, 0, sizeof(Timings));
	}

	Timings& operator+=(Timings t)
	{
		baseline += t.baseline;
		thread += t.thread;
		warp += t.warp;
		block64 += t.block64;
		block128 += t.block128;
		block256 += t.block256;
		block512 += t.block512;
		block1024 += t.block1024;
		device1 += t.device1;
		device2 += t.device2;
		device4 += t.device4;
		device8 += t.device8;
		device16 += t.device16;
		device32 += t.device32;
		return *this;
	}

	Timings& operator/=(int n)
	{
		baseline /= n;
		thread /= n;
		warp /= n;
		block64 /= n;
		block128 /= n;
		block256 /= n;
		block512 /= n;
		block1024 /= n;
		device1 /= n;
		device2 /= n;
		device4 /= n;
		device8 /= n;
		device16 /= n;
		device32 /= n;
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& o, const Timings& t)
	{
		return o << "\n\tBaseline: " << t.baseline << "ms"
			<< "\n\tThread: " << t.thread << "ms"
			<< "\n\tWarp: " << t.warp << "ms"
			<< "\n\tBlock64: " << t.block64 << "ms"
			<< "\n\tBlock128: " << t.block128 << "ms"
			<< "\n\tBlock256: " << t.block256 << "ms"
			<< "\n\tBlock512: " << t.block512 << "ms"
			<< "\n\tBlock1024: " << t.block1024 << "ms"
			<< "\n\tDevice1: " << t.device1 << "ms"
			<< "\n\tDevice2: " << t.device2 << "ms"
			<< "\n\tDevice4: " << t.device4 << "ms"
			<< "\n\tDevice8: " << t.device8 << "ms"
			<< "\n\tDevice16: " << t.device16 << "ms"
			<< "\n\tDevice32: " << t.device32 << "ms";
	}
};

/*
 * Launch benchmarks with template-parameters passed as strings.
 * scalar: "int", "float"
 * axis: "row", "col", "batch"
 * 
 * optimize: skip evaluations when I know that they will be way too slow.
 */
Timings benchmark(cuMat::Index rows, cuMat::Index cols, cuMat::Index batches,
	const std::string& scalar, const std::string& axis, bool compare, bool optimize=false);

#endif