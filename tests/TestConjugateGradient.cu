#include <catch/catch.hpp>

#include <cuMat/Core>
#include <cuMat/src/SimpleRandom.h>
#include <cuMat/IterativeLinearSolvers>

#include "Utils.h"

using namespace cuMat;

TEST_CASE("Conjugate Gradient - Early Out Rhs", "[CG]")
{
    int sizes[] = {5, 10, 50, 100};
    SimpleRandom rand;
    for (int size : sizes) SECTION("Size=" + std::to_string(size)) {
        MatrixXf A(size, size);
        rand.fillUniform(A);
        VectorXf b = VectorXf::Zero(size);

        ConjugateGradient<MatrixXf> cg(A);
        VectorXf x = cg.solve(b);
        REQUIRE(cg.iterations() == 0);
        REQUIRE(cg.error() == 0);
        REQUIRE(x.squaredNorm() == 0);
    }
}

TEST_CASE("Conjugate Gradient - Early Out X", "[CG]")
{
    int sizes[] = {5, 10, 50, 100};
    SimpleRandom rand;
    for (int size : sizes) SECTION("Size=" + std::to_string(size)) {
        MatrixXf A(size, size);
        rand.fillUniform(A);
        VectorXf xTruth(size);
        rand.fillUniform(xTruth);
        VectorXf b = A * xTruth;

        ConjugateGradient<MatrixXf> cg(A);
        VectorXf x = cg.solveWithGuess(b, xTruth);
        REQUIRE(cg.iterations() == 0);
        REQUIRE(cg.error() == Approx(0).margin(1e-10));
        assertMatrixEquality(xTruth, x);
    }
}

TEST_CASE("Conjugate Gradient - Solve Dense", "[CG]")
{
    int sizes[] = {5, 10, 50, 100};
    SimpleRandom rand;
    for (int size : sizes) SECTION("Size=" + std::to_string(size)) {
        MatrixXd A(size, size);
        rand.fillUniform(A);
        A += size * MatrixXd::Identity(size); //make diagonal dominant
        VectorXd xTruth(size);
        rand.fillUniform(xTruth);
        VectorXd b = A * xTruth;

        INFO("A:\n" << A.toEigen());
        INFO("xTruth: " << xTruth.toEigen().transpose());
        INFO("b: " << b.toEigen().transpose());

        ConjugateGradient<MatrixXd> cg(A);
        cg.setMaxIterations(10 * size);
        VectorXd x = cg.solve(b);
		REQUIRE(cg.iterations() > 0);
		REQUIRE(cg.error() > 0);
        REQUIRE(cg.iterations() < cg.maxIterations()); //it should have converged
        REQUIRE(cg.error() <= cg.tolerance());         //the error should be lower than the tolerance
        assertMatrixEqualityRelative(x, xTruth, 1e-4);       //the true solution was found
    }
}

TEST_CASE("Conjugate Gradient - Solve Dense Batched", "[CG]")
{
    constexpr int size = 10;
    constexpr int batches = 3;
    SimpleRandom rand;

    MatrixXd A(size, size);
    rand.fillUniform(A);
    A += size * MatrixXd::Identity(size); //make diagonal dominant
    typedef Matrix<double, Dynamic, 1, batches, ColumnMajor> Vec;
    Vec xTruth(size, 1, batches);
    rand.fillUniform(xTruth);
    Vec b = A * xTruth;

    INFO("A:\n" << A);
    INFO("xTruth: " << xTruth);
    INFO("b: " << b);

    ConjugateGradient<MatrixXd> cg(A);
    cg.setMaxIterations(10 * size);
    Vec x = cg.solve(b);
    REQUIRE(cg.iterations() > 0);
    REQUIRE(cg.error() > 0);
    REQUIRE(cg.iterations() < cg.maxIterations()); //it should have converged
    REQUIRE(cg.error() <= cg.tolerance());         //the error should be lower than the tolerance
    assertMatrixEqualityRelative(x, xTruth, 1e-4);       //the true solution was found
}