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
}