#include "TestLinAlgOps.cuh"
TEST_CASE("dense lin-alg 2", "[Dense]")
{
	SECTION("5x5")
	{
		testlinAlgOps2<5>();
	}
}