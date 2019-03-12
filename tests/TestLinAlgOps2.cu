#include "TestLinAlgOps.cuh"
TEST_CASE("dense lin-alg 2", "[Dense]")
{
	SECTION("3x3")
	{
		testlinAlgOps2<3>();
	}
	SECTION("4x4")
	{
		testlinAlgOps2<4>();
	}
}