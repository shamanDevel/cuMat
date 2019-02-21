#include "TestLinAlgOps.cuh"
TEST_CASE("dense lin-alg 2", "[Dense]")
{
	SECTION("10x10")
	{
	    testlinAlgOps2<10>();
	}
}