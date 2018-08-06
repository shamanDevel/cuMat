#include <catch/catch.hpp>

#include <cuMat/Core>
#include <cuMat/src/ConjugateGradient.h>

using namespace cuMat;

TEST_CASE("Benchmark: Dense Conjugate Gradient", "[Benchmark]")
{
    int size = 1<<12;
    MatrixXf A = MatrixXf::fromEigen((Eigen::MatrixXf::Random(size, size) + (size/2)*Eigen::MatrixXf::Identity(size, size)).eval());
    VectorXf b = VectorXf::fromEigen(Eigen::VectorXf::Random(size).eval());

    ConjugateGradient<MatrixXf> cg(A);
    VectorXf x = cg.solve(b);
    REQUIRE(cg.error() < cg.tolerance()); //converged
}