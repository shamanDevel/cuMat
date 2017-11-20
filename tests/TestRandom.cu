#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"

using namespace cuMat;

TEST_CASE("random", "[random]")
{
    SimpleRandom r;

    SECTION("bool") {
        BMatrixXb m(100, 110, 120);
        r.fillUniform(m, false, true);
        //there is not much we can test here
    }

    SECTION("int")
    {
        BMatrixXi m(100, 110, 120);
        r.fillUniform(m, -10, 50);
        REQUIRE(-10 <= (int)m.minCoeff());
        REQUIRE(50 > (int)m.maxCoeff());
    }

    SECTION("long long")
    {
        BMatrixXll m(100, 110, 120);
        r.fillUniform(m, -100, 500);
        REQUIRE(-100 <= (long long)m.minCoeff());
        REQUIRE(500 > (long long)m.maxCoeff());
    }

    SECTION("float")
    {
        BMatrixXf m(100, 110, 120);
        r.fillUniform(m, 5.5, 12.5);
        REQUIRE(5.5 - 0.00001 <= (float)m.minCoeff());
        REQUIRE(12.5 + 0.00001 > (float)m.maxCoeff());
    }

    SECTION("double")
    {
        BMatrixXd m(100, 110, 120);
        r.fillUniform(m, -5.5, 12.5);
        REQUIRE(-5.5 - 0.00001 <= (double)m.minCoeff());
        REQUIRE(12.5 + 0.00001 > (double)m.maxCoeff());
    }
}