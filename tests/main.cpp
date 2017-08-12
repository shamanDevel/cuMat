#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include <catch/catch.hpp>

#include <cuMat/Matrix.h>

TEST_CASE("dummy")
{
	Matrix<float, 5, 5, 1, 0> m;
}