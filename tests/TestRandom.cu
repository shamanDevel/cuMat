#include <catch/catch.hpp>
#include <limits>

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

    SECTION("complex-float")
    {
        BMatrixXcf m(100, 110, 120);
        r.fillUniform(m, cfloat(5.5, 10.5), cfloat(12.5, 22.5));
        REQUIRE(5.5 - 0.00001 <= (float)m.real().minCoeff());
        REQUIRE(12.5 + 0.00001 > (float)m.real().maxCoeff());
        REQUIRE(10.5 - 0.00001 <= (float)m.imag().minCoeff());
        REQUIRE(22.5 + 0.00001 > (float)m.imag().maxCoeff());
    }

    SECTION("complex-double")
    {
        BMatrixXcd m(100, 110, 120);
        r.fillUniform(m, cdouble(5.5, 10.5), cdouble(12.5, 22.5));
        REQUIRE(5.5 - 0.00001 <= (double)m.real().minCoeff());
        REQUIRE(12.5 + 0.00001 > (double)m.real().maxCoeff());
        REQUIRE(10.5 - 0.00001 <= (double)m.imag().minCoeff());
        REQUIRE(22.5 + 0.00001 > (double)m.imag().maxCoeff());
    }
}

TEST_CASE("random-defaults", "[random]")
{
    SimpleRandom r;

    SECTION("bool") {
        BMatrixXb m(100, 110, 120);
        r.fillUniform(m);
        //there is not much we can test here
    }

    SECTION("int")
    {
        BMatrixXi m(100, 110, 120);
        r.fillUniform(m);
        REQUIRE(0 <= (int)m.minCoeff());
        REQUIRE(std::numeric_limits<int>::max() > (int)m.maxCoeff());
    }

    SECTION("float")
    {
        BMatrixXf m(100, 110, 120);
        r.fillUniform(m);
        REQUIRE(0 - 0.00001 <= (float)m.minCoeff());
        REQUIRE(1 + 0.00001 > (float)m.maxCoeff());
    }

    SECTION("double")
    {
        BMatrixXd m(100, 110, 120);
        r.fillUniform(m);
        REQUIRE(0 - 0.00001 <= (double)m.minCoeff());
        REQUIRE(1 + 0.00001 > (double)m.maxCoeff());
    }

    SECTION("complex-float")
    {
        BMatrixXcf m(100, 110, 120);
        r.fillUniform(m);
        REQUIRE(0 - 0.00001 <= (float)m.real().minCoeff());
        REQUIRE(1 + 0.00001 > (float)m.real().maxCoeff());
        REQUIRE(0 - 0.00001 <= (float)m.imag().minCoeff());
        REQUIRE(1 + 0.00001 > (float)m.imag().maxCoeff());
    }

    SECTION("complex-double")
    {
        BMatrixXcd m(100, 110, 120);
        r.fillUniform(m);
        REQUIRE(0 - 0.00001 <= (double)m.real().minCoeff());
        REQUIRE(1 + 0.00001 > (double)m.real().maxCoeff());
        REQUIRE(0 - 0.00001 <= (double)m.imag().minCoeff());
        REQUIRE(1 + 0.00001 > (double)m.imag().maxCoeff());
    }
}