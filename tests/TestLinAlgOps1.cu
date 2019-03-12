#include "TestLinAlgOps.cuh"
TEST_CASE("dense lin-alg 1", "[Dense]")
{
    SECTION("1x1")
    {
        testlinAlgOps2<1>();
    }
    SECTION("2x2")
    {
        testlinAlgOps2<2>();
    }
}
