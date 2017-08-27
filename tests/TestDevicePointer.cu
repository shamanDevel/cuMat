#include <catch/catch.hpp>

#include <cuMat/src/DevicePointer.h>

//Tests all the different cases in which a device pointer can be used
//All tests pass if Context does not throw an assertion

namespace
{
	void assertMemoryLeak()
	{
		cuMat::Context& context = cuMat::Context::current();
		REQUIRE(context.getAliveDevicePointers() == 0);
		REQUIRE(context.getAliveHostPointers() == 0);
	}
}

TEST_CASE("Copy1", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t1; //constructor
		cuMat::DevicePointer<int> t2 = t1; //constructor [copy]
		cuMat::DevicePointer<int> t3 = t1; //constructor [copy]
		t3 = t2; //assignment [copy]
	}
	assertMemoryLeak();
}

TEST_CASE("Copy2", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t1(16); //constructor
		cuMat::DevicePointer<int> t2 = t1; //constructor [copy]
		cuMat::DevicePointer<int> t3 = t1; //constructor [copy]
		t3 = t2; //assignment [copy]
	}
	assertMemoryLeak();
}

TEST_CASE("Move1", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t1; //constructor
		cuMat::DevicePointer<int> t2; //constructor
		cuMat::DevicePointer<int> t3 = std::move(t1); //constructor [move]
		t3 = std::move(t2); //assignment [move]
	}
	assertMemoryLeak();
}

TEST_CASE("Move2", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t1(16); //constructor
		cuMat::DevicePointer<int> t2(16); //constructor
		cuMat::DevicePointer<int> t3 = std::move(t1); //constructor [move]
		t3 = std::move(t2); //assignment [move]
	}
	assertMemoryLeak();
}

cuMat::DevicePointer<int> createPointer1()
{
	cuMat::DevicePointer<int> t;
	return t;
}

cuMat::DevicePointer<int> createPointer2()
{
	cuMat::DevicePointer<int> t(16);
	return t;
}

TEST_CASE("Move3", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t = createPointer1(); //constructor \n constructor [move]
		t = createPointer1(); //constructor \n assignment [move]
	}
	assertMemoryLeak();
}

TEST_CASE("Move4", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t = createPointer2(); //constructor \n constructor [move]
		t = createPointer1(); //constructor \n assignment [move]
	}
	assertMemoryLeak();
}

TEST_CASE("Casting", "[device_pointer]")
{
	{
		cuMat::DevicePointer<int> t1(8);
		const cuMat::DevicePointer<int> t2(8);
		REQUIRE(t1.pointer() != nullptr);
		REQUIRE(t2.pointer() != nullptr);
	}
	assertMemoryLeak();
}
