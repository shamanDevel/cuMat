#include "TestLinAlgOps.cuh"
TEST_CASE("dense lin-alg 2", "[Dense]")
{
	SECTION("4x4")
	{
		testlinAlgOps2<4>();
	}
	SECTION("5x5")
	{
		testlinAlgOps2<5>();
	}
	SECTION("10x10")
	{
	    testlinAlgOps2<10>();
	}
}