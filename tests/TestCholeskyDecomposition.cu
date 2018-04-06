#include <catch/catch.hpp>

#include <cuMat/Dense>
#include "Utils.h"

using namespace cuMat;

template<typename Scalar, int Flags>
void testCholeskyDecompositionReal()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, 2, Flags> mat_t;
    double dataA[2][5][5] {
        {
            {2.1383,1.34711,2.44672,1.17769,1.27679},
            {1.34711,1.74138,2.11905,0.850403,1.4032},
            {2.44672,2.11905,3.555,1.7608,2.06206},
            {1.17769,0.850403,1.7608,1.42897,1.14107},
            {1.27679,1.4032,2.06206,1.14107,1.4566}
        },
        {
            {1.28668,1.01275,0.833612,1.41058,0.582176},
            {1.01275,2.23712,1.46201,1.4821,1.17532},
            {0.833612,1.46201,1.17368,1.33016,0.744467},
            {1.41058,1.4821,1.33016,1.88637,0.773026},
            {0.582176,1.17532,0.744467,0.773026,0.730395}
        }
    };
    mat_t A = BMatrixXdR::fromArray(dataA).cast<Scalar>().template block<5, 5, 2>(0, 0, 0);

    double dataB[2][5][2] {
        { 
            { 0.352364, 1.86783 },
            { 0.915126, -0.78421 },
            { -1.71784, -1.47416 },
            { -1.84341, - 0.58641 },
            { 0.210527, 0.928482 } 
        },
        { 
            { 0.0407573, 0.219543 },
            { 0.748412, 0.564233 },
            { 1.41703, 1.85561 },
            { -0.897485, 0.418297 },
            { 1.682, -0.303229 } 
        }
    };
    mat_t B = BMatrixXdR::fromArray(dataB).cast<Scalar>().template block<5, 2, 2>(0, 0, 0);

    double dataAB[2][5][2] {
        {
            {6.11211,10.4349},
            {1.27924,-7.19059},
            {-8.95239,-12.0052},
            {-2.75267,-5.99761},
            {8.38455,20.1113}
        },
        {
            {260.821,208.526},
            {-134.634,-100.712},
            {450.214,359.579},
            {-394.991,-314.458},
            {-29.7862,-38.2601}
        }
    };
    mat_t AB = BMatrixXdR::fromArray(dataAB).cast<Scalar>().template block<5, 2, 2>(0, 0, 0);

    double determinantData[2][1][1]{
        {{0.0359762}},
        {{0.000185246}}
    };
    mat_t determinant = BMatrixXdR::fromArray(determinantData).cast<Scalar>().template block<1, 1, 2>(0, 0, 0);

    //perform Cholesky decomposition
    CholeskyDecomposition<mat_t> decomposition(A);
    typename CholeskyDecomposition<mat_t>::EvaluatedMatrix matrixCholesky = decomposition.getMatrixCholesky();
    REQUIRE(A.data() != matrixCholesky.data()); //ensure that the original matrix was not modified

    //Solve linear system
    SECTION("solve") {
        auto solveResult = decomposition.solve(B).eval();
        INFO("input matrix:\n" << A);
        INFO("decomposition:\n" << matrixCholesky);
        INFO("A*X = " << (A*solveResult).eval());
        assertMatrixEqualityRelative(AB, solveResult, 1e-2);
    }
    
    //compute determinant
    SECTION("det") {
        auto determinantResult = decomposition.determinant().eval();
        assertMatrixEqualityRelative(determinant, determinantResult, 1e-2);
    }
    
    //compute determinant
    SECTION("log-det") {
        auto logDeterminantResult = decomposition.logDeterminant().eval();
        assertMatrixEqualityRelative(determinant.cwiseLog().eval(), logDeterminantResult, 1e-2);
    }

    //Test inverse
    SECTION("inverse") {
        auto inverseResult = decomposition.inverse().eval();
        INFO("inverse: \n" << inverseResult);
        assertMatrixEquality(mat_t::Identity(5, 5, 2), A*inverseResult, 1e-2);
    }
}

/*
template<typename Scalar, int Flags>
void testLUDecompositionComplex()
{
#define I *cdouble(0,1)

    typedef Matrix<Scalar, Dynamic, Dynamic, 2, Flags> mat_t;
    cdouble dataA[2][5][5]{
        { 
            { -0.87232 - 0.935388 I, 1.01768 - 1.89061 I, 1.99212 + 0.702465 I, 0.891497 + 1.46058 I, 0.973509 - 0.726869 I },
            { -1.7624 + 0.115476 I, -1.28832 - 1.1484 I, -0.251518 + 0.977168 I, 0.928271 - 1.20425 I, 1.30158 - 0.329899 I },
            { -0.0111449 + 1.8512 I, 0.218638 + 0.357781 I, 0.436318 + 0.458631 I, 0.300155 - 0.90286 I, 1.84898 + 1.24462 I },
            { 1.29907 - 1.46868 I, -0.775703 - 1.11854 I, 1.28314 + 1.76659 I, -1.20293 + 1.90851 I, 0.493409 - 0.35713 I },
            { 1.91903 + 1.27134 I, 0.116577 + 1.00647 I, 1.70334 - 1.27062 I, 1.75497 - 1.00864 I, 0.629163 + 1.45738 I } 
        },
        { 
            { -1.18698 + 1.57933 I, 1.4197 + 0.138538 I, -0.106781 + 0.603585 I, 1.70241 + 0.942044 I, -1.58697 - 1.22207 I },
            { 1.57946 + 0.504369 I, 1.88721 - 1.09034 I, 0.592362 + 1.09007 I, 1.45566 - 0.300413 I, 1.3647 + 0.802899 I },
            { 0.489904 - 0.819587 I, -1.47237 - 1.60618 I, -1.66062 - 1.38051 I, 0.567336 - 1.78629 I, -1.31663 - 0.5528 I },
            { -1.9819 + 0.757709 I, -1.44771 - 0.219543 I, -0.428032 - 0.922055 I, -0.420926 + 1.6984 I, 0.327754 + 0.0803375 I },
            { 0.951035 - 0.640323 I, -0.0593802 + 1.79482 I, 0.153952 - 0.512452 I, 1.63995 - 1.15083 I, -1.35505 - 1.60575 I } 
        }
    };
    mat_t A = BMatrixXcdR::fromArray(dataA).cast<Scalar>().template block<5, 5, 2>(0, 0, 0);

    cdouble dataB[2][5][2]{
        { 
            { -1.25871 + 1.09482 I, 0.914983 - 0.89768 I },
            { -0.871042 - 1.83597 I, 1.1436 + 0.0848052 I },
            { 1.7451 + 1.85383 I, 0.684588 - 0.430585 I },
            { 0.353774 - 1.18935 I, -1.29296 - 0.546456 I },
            { -1.07524 + 0.317155 I, 0.385134 + 0.560481 I } 
        },
        { 
            { -1.7115 + 0.242376 I, -1.11541 + 0.211799 I },
            { -0.939136 + 0.250557 I, -1.90742 + 1.41402 I },
            { -1.07052 - 0.839619 I, -0.291077 - 0.223146 I },
            { 0.259481 - 1.19091 I, -0.0600881 + 1.37227 I },
            { 0.0625118 - 0.612875 I, 1.58531 - 0.581563 I } 
        }
    };
    mat_t B = BMatrixXcdR::fromArray(dataB).cast<Scalar>().template block<5, 2, 2>(0, 0, 0);

    cdouble dataAB[2][5][2]{
        { 
            { 0.308061 - 1.45214 I, -0.299341 + 0.273846 I },
            { -0.869667 + 0.539718 I, 0.695494 - 0.11568 I },
            { -1.78851 + 1.63544 I, 0.386792 - 0.592088 I },
            { -0.350786 - 2.48615 I, 0.251472 + 0.954549 I },
            { 1.7374 - 0.0896004 I, -0.161408 + 0.0678378 I } 
        },
        { 
            { 0.65992 + 0.261577 I, -0.0400807 - 1.45049 I },
            { -0.249235 - 0.163502 I, -0.208349 - 0.77229 I },
            { 0.784653 + 0.544065 I, 0.52958 + 0.767102 I },
            { -0.398583 - 0.711315 I, -0.918782 + 1.38866 I },
            { -0.382484 - 0.302958 I, 0.576564 + 0.852561 I } 
        }
    };
    mat_t AB = BMatrixXcdR::fromArray(dataAB).cast<Scalar>().template block<5, 2, 2>(0, 0, 0);

    cdouble determinantData[2][1][1]{
        { { 131.186 + 58.9019 I } },
        { { -86.5874 + 23.9651 I } }
    };
    mat_t determinant = BMatrixXcdR::fromArray(determinantData).cast<Scalar>().template block<1, 1, 2>(0, 0, 0);

#undef I

    //perform LU decomposition
    LUDecomposition<mat_t> decomposition(A);
    typename LUDecomposition<mat_t>::EvaluatedMatrix matrixLU = decomposition.getMatrixLU();
    typename LUDecomposition<mat_t>::PivotArray pivots = decomposition.getPivots();

    REQUIRE(A.data() != matrixLU.data()); //ensure that the original matrix was not modified

    //Solve linear system
    auto solveResult = decomposition.solve(B).eval();
    INFO("input matrix:\n" << A);
    INFO("decomposition:\n" << matrixLU);
    INFO("pivots:\n" << pivots);
    INFO("A*X = " << (A*solveResult).eval());
    assertMatrixEquality(AB, solveResult, 1e-5);

    //compute determinant
    //TODO: for some reasons, the determinant computation fails with a compiler error for complex values
    //auto determinantResult = decomposition.determinant().eval();
    //assertMatrixEquality(determinant, determinantResult, 1e-3);

    //Test inverse
    auto inverseResult = decomposition.inverse().eval();
    INFO("inverse: \n" << inverseResult);
    assertMatrixEquality(mat_t::Identity(5, 5, 2), A*inverseResult, 1e-5);
}
*/

TEST_CASE("Cholesky-Decomposition", "[Dense]")
{
    SECTION("float")
    {
        SECTION("row major")
        {
            testCholeskyDecompositionReal<float, RowMajor>();
        }
        SECTION("column major")
        {
            testCholeskyDecompositionReal<float, ColumnMajor>();
        }
    }
    SECTION("double")
    {
        SECTION("row major")
        {
            testCholeskyDecompositionReal<double, RowMajor>();
        }
        SECTION("column major")
        {
            testCholeskyDecompositionReal<double, ColumnMajor>();
        }
    }
    /*
    SECTION("complex-float")
    {
        SECTION("row major")
        {
            testLUDecompositionComplex<cfloat, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecompositionComplex<cfloat, ColumnMajor>();
        }
    }
    SECTION("complex-double")
    {
        SECTION("row major")
        {
            testLUDecompositionComplex<cdouble, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecompositionComplex<cdouble, ColumnMajor>();
        }
    }
    */
}

