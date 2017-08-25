#include <catch/catch.hpp>

#include <thread>
#include <mutex>
#include <set>
#include <vector>

#include <cuMat/src/Context.h>
#include "Barrier.h"

TEST_CASE("single_context", "[context]")
{
	//Test if the context can be created
	cuMat::Context& context = cuMat::Context::current();
	REQUIRE(context.stream() != nullptr);
}

TEST_CASE("muliple_contexts", "[context]")
{
	//Test if multiple contexts can be created
	int count = 8;
	std::mutex mutex1;
	Barrier barrier(count);
	std::set<cudaStream_t> streams;

	//create threads
	std::vector<std::thread> threads;
	for (int i=0; i<count; ++i)
	{
		threads.push_back(std::thread([&]()
		{
			cuMat::Context& context = cuMat::Context::current();
			REQUIRE(context.stream() != nullptr);
			mutex1.lock();
			//std::cout << context.stream() << std::endl;
			streams.insert(context.stream());
			mutex1.unlock();

			barrier.Wait();
		}));
	}

	//wait for threads to terminate
	for (int i=0; i<count; ++i)
	{
		threads[i].join();
	}

	//test if really count-many different streams were created
	REQUIRE(streams.size() == count);
}

TEST_CASE("allocator", "[context]")
{
	//Test if the allocator works
	cuMat::Context& context = cuMat::Context::current();
	void* mem;

	//freeing of NULL does nothing
	context.freeDevice(nullptr);
	context.freeHost(nullptr);

	//allocating size zero works
	mem = context.mallocDevice(0);
	context.freeDevice(mem);
	mem = context.mallocHost(0);
	context.freeHost(mem);

	//allocate some memory
	std::vector<size_t> sizes({ 1,4,16,1024,1000000 });
	for (size_t s : sizes)
	{
		mem = context.mallocHost(s);
		REQUIRE(mem != nullptr);
		context.freeHost(mem);
		mem = context.mallocDevice(s);
		REQUIRE(mem != nullptr);
		context.freeDevice(mem);
	}

	REQUIRE(context.getAliveDevicePointers() == 0);
	REQUIRE(context.getAliveHostPointers() == 0);
}