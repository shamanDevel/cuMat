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
		typedef SparsityPattern<SparseFlags::CSR> SPattern;
		SPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 1, 1, 2, 0, 3, 4, 2, 4).finished());
        pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(5) << 0, 2, 4, 7, 9).finished());
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
		typedef SparsityPattern<SparseFlags::CSC> SPattern;
		SPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 2, 0, 1, 1, 3, 2, 2, 3).finished());
        pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(6) << 0, 2, 4, 6, 7, 9).finished());
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

	SECTION("ELLPACK")
	{
		//create sparse matrix
		typedef SparseMatrix<float, 2, SparseFlags::ELLPACK> SMatrix_t;
		typedef SparsityPattern<SparseFlags::ELLPACK> SPattern;
		SPattern pattern;
		pattern.rows = 4;
		pattern.cols = 5;
		pattern.nnzPerRow = 3;
		pattern.indices = SPattern::IndexMatrix::fromEigen((Eigen::MatrixXi(4,3) << 
			0, 1, -1,
			1, 2, -1,
			0, 3, 4,
			2, 4, -1).finished());
		REQUIRE_NOTHROW(pattern.assertValid());
		SMatrix_t smatrix(pattern);
		smatrix.getData().slice(0) = MatrixXf::fromEigen((Eigen::MatrixXf(4,3) << 
			1, 4, 666,
			2, 3, 666,
			5, 7, 8,
			9, 6, 666).finished());
		smatrix.getData().slice(1) = MatrixXf::fromEigen((Eigen::MatrixXf(4, 3) <<
			-1, -4, 666,
			-2, -3, 666,
			-5, -7, -8,
			-9, -6, 666).finished());
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
		typedef SparsityPattern<SparseFlags::CSR> SPattern;
		SPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 1, 1, 2, 0, 3, 4, 2, 4).finished());
        pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(5) << 0, 2, 4, 7, 9).finished());
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
		typedef SparsityPattern<SparseFlags::CSC> SPattern;
		SPattern pattern;
        pattern.rows = 4;
        pattern.cols = 5;
        pattern.nnz = 9;
        pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 2, 0, 1, 1, 3, 2, 2, 3).finished());
        pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(6) << 0, 2, 4, 6, 7, 9).finished());
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

	SECTION("ELLPACK")
	{
		//create sparse matrix
		typedef SparseMatrix<float, 2, SparseFlags::ELLPACK> SMatrix_t;
		typedef SparsityPattern<SparseFlags::ELLPACK> SPattern;
		SPattern pattern;
		pattern.rows = 4;
		pattern.cols = 5;
		pattern.nnzPerRow = 3;
		pattern.indices = SPattern::IndexMatrix::fromEigen((Eigen::MatrixXi(4, 3) <<
			0, 1, -1,
			1, 2, -1,
			0, 3, 4,
			2, 4, -1).finished());
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
		MatrixXf batch1 = MatrixXf::fromEigen((Eigen::MatrixXf(4, 3) <<
			1, 4, 0,
			2, 3, 0,
			5, 7, 8,
			9, 6, 0).finished());
		MatrixXf batch2 = MatrixXf::fromEigen((Eigen::MatrixXf(4, 3) <<
			-1, -4, 0,
			-2, -3, 0,
			-5, -7, -8,
			-9, -6, 0).finished());
		MatrixXb mask = MatrixXb::fromEigen((Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>(4, 3) <<
			true, true, false,
			true, true, false,
			true, true, true,
			true, true, false).finished());
		assertMatrixEquality(smatrix1.getData().slice(0), batch1, mask);
		assertMatrixEquality(smatrix1.getData().slice(1), batch2, mask);
		assertMatrixEquality(smatrix2.getData().slice(0), batch1, mask);
		assertMatrixEquality(smatrix2.getData().slice(1), batch2, mask);

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

TEST_CASE("Sparse-Cwise", "[Sparse]")
{
    //create source sparse matrix
    typedef SparseMatrix<float, 1, SparseFlags::CSC> SMatrix_t;
	typedef SparsityPattern<SparseFlags::CSC> SPattern;
	SPattern pattern;
    pattern.rows = 4;
    pattern.cols = 5;
    pattern.nnz = 9;
    pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(9) << 0, 2, 0, 1, 1, 3, 2, 2, 3).finished());
    pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(6) << 0, 2, 4, 6, 7, 9).finished());
    REQUIRE_NOTHROW(pattern.assertValid());
    SMatrix_t smatrix(pattern);
    smatrix.getData().slice(0) = VectorXf::fromEigen((Eigen::VectorXf(9) << 1, 5, -4, 2, -3, 9, 7, -8, 6).finished());

    //create target sparse matrix of the same pattern, but uninitialized
    SMatrix_t tmatrix1(pattern);
    SMatrix_t tmatrix2(pattern);

    //some c-wise operations
    tmatrix1 = SMatrix_t::Identity(4, 5) + smatrix.cwiseAbs2();
    tmatrix2 = SMatrix_t::Identity(4, 5) + smatrix.direct().cwiseAbs2();

    float expectedDense[1][4][5] = {
            {
                {2, 16, 0, 0, 0},
                {0, 5, 9, 0, 0},
                {25, 0, 0/*not 1*/, 49, 64},
                {0, 0, 81, 0/*not 1*/, 36}
            },
    };
    assertMatrixEquality(expectedDense, tmatrix1);
    assertMatrixEquality(expectedDense, tmatrix2);
}

TEST_CASE("Compound-SparseMatrix", "[Sparse]")
{
    //int data[1][2][3] = { {{1, 2, 0}, {0, 3, 4}} };
    typedef SparseMatrix<int, 1, SparseFlags::CSR> SMatrix_t;
	typedef SparsityPattern<SparseFlags::CSR> SPattern;
	SPattern pattern;
    pattern.rows = 2;
    pattern.cols = 3;
    pattern.nnz = 4;
    pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(4) << 0, 1, 1, 2).finished());
    pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(3) << 0, 2, 4).finished());
    REQUIRE_NOTHROW(pattern.assertValid());
    SMatrix_t mat1(pattern);
    SMatrix_t mat2(pattern);
    mat1.getData().slice(0) = VectorXi::fromEigen((Eigen::VectorXi(4) << 1, 2, 3, 4).finished());
    
    SECTION("add") {
        mat2 = mat1.deepClone();
        mat2 += mat1;
        int expected1[1][2][3] = { { { 2, 4, 0 },{ 0, 6, 8 } } };
        assertMatrixEquality(expected1, mat2);
    }

    SECTION("sub") {
        mat2 = Matrix<int, 2, 3, 1, ColumnMajor>::Zero();
        mat2 -= mat1;
        int expected2[1][2][3] = { { { -1, -2, 0 },{ 0, -3, -4 } } };
        assertMatrixEquality(expected2, mat2);
    }

    SECTION("mod") {
        mat2 = mat1 + 3;
        mat2 %= mat1;
        int expected3[1][2][3] = { { { 0, 1, 0 },{ 0, 0, 3 } } };
        assertMatrixEquality(expected3, mat2);
    }

    SECTION("mul - scalar")
    {
        mat2 = mat1.deepClone();
        mat2 *= 2;
        int expected4[1][2][3] = { { { 2, 4, 0 },{ 0, 6, 8 } } };
        assertMatrixEquality(expected4, mat2);
    }
}

TEST_CASE("Compound-SparseMatrix-Direct", "[Sparse]")
{
	//int data[1][2][3] = { {{1, 2, 0}, {0, 3, 4}} };
	typedef SparseMatrix<int, 1, SparseFlags::CSR> SMatrix_t;
	typedef SparsityPattern<SparseFlags::CSR> SPattern;
	SPattern pattern;
	pattern.rows = 2;
	pattern.cols = 3;
	pattern.nnz = 4;
	pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(4) << 0, 1, 1, 2).finished());
	pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(3) << 0, 2, 4).finished());
	REQUIRE_NOTHROW(pattern.assertValid());
	SMatrix_t mat1(pattern);
	SMatrix_t mat2(pattern);
	mat1.getData().slice(0) = VectorXi::fromEigen((Eigen::VectorXi(4) << 1, 2, 3, 4).finished());

	SECTION("add") {
		mat2 = mat1.deepClone();
		mat2 += mat1.direct();
		int expected1[1][2][3] = { { { 2, 4, 0 },{ 0, 6, 8 } } };
		assertMatrixEquality(expected1, mat2);
	}

	SECTION("sub") {
		mat2 = Matrix<int, 2, 3, 1, ColumnMajor>::Zero();
		mat2 -= mat1.direct();
		int expected2[1][2][3] = { { { -1, -2, 0 },{ 0, -3, -4 } } };
		assertMatrixEquality(expected2, mat2);
	}

	SECTION("mod") {
		mat2 = mat1 + 3;
		mat2 %= mat1.direct();
		int expected3[1][2][3] = { { { 0, 1, 0 },{ 0, 0, 3 } } };
		assertMatrixEquality(expected3, mat2);
	}

	SECTION("mul - scalar")
	{
		mat2 = mat1.deepClone();
		mat2 *= 2;
		int expected4[1][2][3] = { { { 2, 4, 0 },{ 0, 6, 8 } } };
		assertMatrixEquality(expected4, mat2);
	}
}
