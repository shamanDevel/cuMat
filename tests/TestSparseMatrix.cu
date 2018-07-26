#include <catch/catch.hpp>

#include <cuMat/Sparse>

#include "Utils.h"

using namespace cuMat;

//Assigns a sparse matrix to a dense matrix
TEST_CASE("Sparse -> Dense", "[Sparse]")
{
    SECTION("CSR")
    {
        //create sparse matrix
        typedef SparseMatrix<float, 2, SparseFlags::CSR> SMatrix_t;
        SMatrix_t::SparsityPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 1, 1, 2, 0, 3, 4, 2, 4).finished());
        pattern.JA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(5) << 0, 2, 4, 7, 9).finished());
        REQUIRE_NOTHROW(pattern.assertValid());
        SMatrix_t smatrix(pattern);
        smatrix.getData().slice(0) = VectorXf::fromEigen((Eigen::VectorXf(9) << 1, 4, 2, 3, 5, 7, 8, 9, 6).finished());
        smatrix.getData().slice(1) = VectorXf::fromEigen((Eigen::VectorXf(9) << -1, -4, -2, -3, -5, -7, -8, -9, -6).finished());
        INFO(smatrix);

        //assign to dense matrix
        BMatrixXfC mat1 = smatrix;
        BMatrixXfR mat2 = smatrix;

        //Test if they are equal
        float expected[2][4][5] = {
            {
                {1, 4, 0, 0, 0},
                {0, 2, 3, 0, 0},
                {5, 0, 0, 7, 8},
                {0, 0, 9, 0, 6}
            },
            {
                {-1, -4, 0, 0, 0},
                {0, -2, -3, 0, 0},
                {-5, 0, 0, -7, -8},
                {0, 0, -9, 0, -6}
            }
        };
        assertMatrixEquality(expected, mat1);
        assertMatrixEquality(expected, mat2);
    }

    SECTION("CSC")
    {
        //create sparse matrix
        typedef SparseMatrix<float, 2, SparseFlags::CSC> SMatrix_t;
        SMatrix_t::SparsityPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 2, 0, 1, 1, 3, 2, 2, 3).finished());
        pattern.JA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(6) << 0, 2, 4, 6, 7, 9).finished());
        REQUIRE_NOTHROW(pattern.assertValid());
        SMatrix_t smatrix(pattern);
        smatrix.getData().slice(0) = VectorXf::fromEigen((Eigen::VectorXf(9) << 1, 5, 4, 2, 3, 9, 7, 8, 6).finished());
        smatrix.getData().slice(1) = VectorXf::fromEigen((Eigen::VectorXf(9) << -1, -5, -4, -2, -3, -9, -7, -8, -6).finished());
        INFO(smatrix);

        //assign to dense matrix
        Profiling::instance().resetAll();
        BMatrixXfC mat1 = smatrix;
        REQUIRE(Profiling::instance().getReset(Profiling::EvalCwise) == 1);
        REQUIRE(Profiling::instance().getReset(Profiling::EvalAny) == 1);
        BMatrixXfR mat2 = smatrix;
        REQUIRE(Profiling::instance().getReset(Profiling::EvalCwise) == 1);
        REQUIRE(Profiling::instance().getReset(Profiling::EvalAny) == 1);

        //Test if they are equal
        float expected[2][4][5] = {
            {
                {1, 4, 0, 0, 0},
                {0, 2, 3, 0, 0},
                {5, 0, 0, 7, 8},
                {0, 0, 9, 0, 6}
            },
            {
                {-1, -4, 0, 0, 0},
                {0, -2, -3, 0, 0},
                {-5, 0, 0, -7, -8},
                {0, 0, -9, 0, -6}
            }
        };
        assertMatrixEquality(expected, mat1);
        assertMatrixEquality(expected, mat2);
    }
}

//Assigns a sparse matrix to a dense matrix
TEST_CASE("Dense -> Sparse", "[Sparse]")
{
    SECTION("CSR")
    {
        //create sparse matrix
        typedef SparseMatrix<float, 2, SparseFlags::CSR> SMatrix_t;
        SMatrix_t::SparsityPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 1, 1, 2, 0, 3, 4, 2, 4).finished());
        pattern.JA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(5) << 0, 2, 4, 7, 9).finished());
        REQUIRE_NOTHROW(pattern.assertValid());
        SMatrix_t smatrix1(pattern);
        SMatrix_t smatrix2(pattern);

        //Input Dense data
        float inputData[2][4][5] = {
            {
                {1, 4, 42, 42, 42},
                {42, 2, 3, 42, 42},
                {5, 42, 42, 7, 8},
                {42, 42, 9, 42, 6}
            },
            {
                {-1, -4, 42, 42, 42},
                {42, -2, -3, 42, 42},
                {-5, 42, 42, -7, -8},
                {42, 42, -9, 42, -6}
            }
        };
        BMatrixXfR mat1 = BMatrixXfR::fromArray(inputData);
        BMatrixXfC mat2 = mat1.deepClone<ColumnMajor>();

        //assign dense to sparse
        Profiling::instance().resetAll();
        smatrix1 = mat1;
        REQUIRE(Profiling::instance().getReset(Profiling::EvalCwiseSparse) == 1);
        REQUIRE(Profiling::instance().getReset(Profiling::EvalAny) == 1);
        smatrix2 = mat2;
        REQUIRE(Profiling::instance().getReset(Profiling::EvalCwiseSparse) == 1);
        REQUIRE(Profiling::instance().getReset(Profiling::EvalAny) == 1);

        //Check data array
        VectorXf batch1 = VectorXf::fromEigen((Eigen::VectorXf(9) << 1, 4, 2, 3, 5, 7, 8, 9, 6).finished());
        VectorXf batch2 = VectorXf::fromEigen((Eigen::VectorXf(9) << -1, -4, -2, -3, -5, -7, -8, -9, -6).finished());
        assertMatrixEquality(smatrix1.getData().slice(0), batch1);
        assertMatrixEquality(smatrix1.getData().slice(1), batch2);
        assertMatrixEquality(smatrix2.getData().slice(0), batch1);
        assertMatrixEquality(smatrix2.getData().slice(1), batch2);

        //Test directly with asserMatrixEquality
        //Uses Cwise-Read to extract the matrix blocks / slices
        float expectedDense[2][4][5] = {
            {
                {1, 4, 0, 0, 0},
                {0, 2, 3, 0, 0},
                {5, 0, 0, 7, 8},
                {0, 0, 9, 0, 6}
            },
            {
                {-1, -4, 0, 0, 0},
                {0, -2, -3, 0, 0},
                {-5, 0, 0, -7, -8},
                {0, 0, -9, 0, -6}
            }
        };
        assertMatrixEquality(expectedDense, smatrix1);
        assertMatrixEquality(expectedDense, smatrix2);
    }

    SECTION("CSC")
    {
        //create sparse matrix
        typedef SparseMatrix<float, 2, SparseFlags::CSC> SMatrix_t;
        SMatrix_t::SparsityPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 2, 0, 1, 1, 3, 2, 2, 3).finished());
        pattern.JA = SMatrix_t::IndexVector::fromEigen((Eigen::VectorXi(6) << 0, 2, 4, 6, 7, 9).finished());
        REQUIRE_NOTHROW(pattern.assertValid());
        SMatrix_t smatrix1(pattern);
        SMatrix_t smatrix2(pattern);

        //Input Dense data
        float inputData[2][4][5] = {
            {
                {1, 4, 42, 42, 42},
                {42, 2, 3, 42, 42},
                {5, 42, 42, 7, 8},
                {42, 42, 9, 42, 6}
            },
            {
                {-1, -4, 42, 42, 42},
                {42, -2, -3, 42, 42},
                {-5, 42, 42, -7, -8},
                {42, 42, -9, 42, -6}
            }
        };
        BMatrixXfR mat1 = BMatrixXfR::fromArray(inputData);
        BMatrixXfC mat2 = mat1.deepClone<ColumnMajor>();

        //assign dense to sparse
        Profiling::instance().resetAll();
        smatrix1 = mat1;
        REQUIRE(Profiling::instance().getReset(Profiling::EvalCwiseSparse) == 1);
        REQUIRE(Profiling::instance().getReset(Profiling::EvalAny) == 1);
        smatrix2 = mat2;
        REQUIRE(Profiling::instance().getReset(Profiling::EvalCwiseSparse) == 1);
        REQUIRE(Profiling::instance().getReset(Profiling::EvalAny) == 1);

        //Check data array
        VectorXf batch1 = VectorXf::fromEigen((Eigen::VectorXf(9) << 1, 5, 4, 2, 3, 9, 7, 8, 6).finished());
        VectorXf batch2 = VectorXf::fromEigen((Eigen::VectorXf(9) << -1, -5, -4, -2, -3, -9, -7, -8, -6).finished());
        assertMatrixEquality(smatrix1.getData().slice(0), batch1);
        assertMatrixEquality(smatrix1.getData().slice(1), batch2);
        assertMatrixEquality(smatrix2.getData().slice(0), batch1);
        assertMatrixEquality(smatrix2.getData().slice(1), batch2);

        //Test directly with asserMatrixEquality
        //Uses Cwise-Read to extract the matrix blocks / slices
        float expectedDense[2][4][5] = {
            {
                {1, 4, 0, 0, 0},
                {0, 2, 3, 0, 0},
                {5, 0, 0, 7, 8},
                {0, 0, 9, 0, 6}
            },
            {
                {-1, -4, 0, 0, 0},
                {0, -2, -3, 0, 0},
                {-5, 0, 0, -7, -8},
                {0, 0, -9, 0, -6}
            }
        };
        assertMatrixEquality(expectedDense, smatrix1);
        assertMatrixEquality(expectedDense, smatrix2);
    }
}