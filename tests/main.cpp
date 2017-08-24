#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include <catch/catch.hpp>

#include <cuMat/src/Matrix.h>

TEST_CASE("dummy")
{
	cuMat::Matrix<float, 5, 5, 1, 0> m;
}