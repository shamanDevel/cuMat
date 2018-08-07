#include <catch/catch.hpp>

#include <cuMat/Core>
#include <cuMat/Sparse>
#include "Utils.h"

using namespace cuMat;

template<typename Scalar>
void testSparseOuterProduct()
{
    typedef SparseMatrix<Scalar, 1, SparseFlags::CSR> SMatrix_t;
    typename SMatrix_t::SparsityPattern pattern;
    pattern.rows = 5;
    pattern.cols = 5;
    pattern.nnz = 13;
    pattern.IA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(13) << 0,1, 0,1,2, 1,2,3, 2,3,4, 3,4).finished());
    pattern.JA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(6) << 0, 2, 5, 8, 11, 13).finished());
    REQUIRE_NOTHROW(pattern.assertValid());
    SMatrix_t smatrix1(pattern);
    SMatrix_t smatrix2(pattern);
    Eigen::Matrix<Scalar, Dynamic, Dynamic> mask(5, 5); 
    mask << 1,1,0,0,0,
            1,1,1,0,0,
            0,1,1,1,0,
            0,0,1,1,1,
            0,0,0,1,1;

    typedef Matrix<Scalar, Dynamic, 1, 1, ColumnMajor> Vector_t;
    typedef Eigen::Matrix<Scalar, Dynamic, 1, Eigen::StorageOptions::ColMajor> EVector_t;
    EVector_t ev1 = (EVector_t(5) << -10, 2, 7, -3, 4).finished();
    EVector_t ev2 = (EVector_t(5) << 8, 4, 8, -6, 3).finished();
    Vector_t v1 = Vector_t::fromEigen(ev1);
    Vector_t v2 = Vector_t::fromEigen(ev2);

#define RESET_PROFILING Profiling::instance().resetAll()
#define CHECK_PROFILING \
    REQUIRE(Profiling::instance().get(Profiling::Counter::EvalAny) == 1); \
    REQUIRE(Profiling::instance().get(Profiling::Counter::EvalCwiseSparse) == 1); \
    Profiling::instance().resetAll();

    RESET_PROFILING;
    smatrix1 = v1 * v2.transpose(); //outer product
    CHECK_PROFILING

    //it has to be really cwise
    RESET_PROFILING;
    smatrix2 = ((v1 + 0) * (v2.transpose() + 0)) + 0; //outer product
    CHECK_PROFILING

    //verify the result
    Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::StorageOptions::ColMajor> etruth = (ev1 * ev2.transpose()).cwiseProduct(mask);
    Matrix<Scalar, Dynamic, Dynamic, 1, ColumnMajor> truth = Matrix<Scalar, Dynamic, Dynamic, 1, ColumnMajor>::fromEigen(etruth);
    assertMatrixEquality(truth, smatrix1);
    assertMatrixEquality(truth, smatrix2);

#undef RESET_PROFILING
#undef CHECK_PROFILING
}

TEST_CASE("Sparse (cwise) outer product", "[Sparse]")
{
    SECTION("int") {testSparseOuterProduct<int>();}
    SECTION("float") {testSparseOuterProduct<float>();}
    SECTION("double") {testSparseOuterProduct<double>();}
}