#include <catch/catch.hpp>

#include <cuMat/Dense>
#include "Utils.h"

using namespace cuMat;

template<typename Scalar, int Flags>
void testLUDecompositionReal()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, 2, Flags> mat_t;
    double dataA[2][5][5] {
        { 
            { -0.509225, -0.713714, -0.376735, 1.50941, -1.51876 },
            { -0.394598, 0.740255, 1.52784, -1.79412, 0.915032 },
            { -0.889638, 0.697614, -1.53048, -0.78504, 0.470366 },
            { 0.254883, 1.82631, -0.110123, -0.143651, 1.34646 },
            { -1.50108, -1.51388, 1.19646, -0.127689, 1.96073 } 
        },
        { 
            { 1.63504, -0.127594, -1.65697, -1.13212, -1.34848 },
            { -1.20512, 0.799606, 0.399986, -0.194832, -1.6951 },
            { 1.37783, -1.62132, -0.064481, 1.43579, -0.772237 },
            { 1.63069, 1.2503, 0.0430382, -1.32802, -1.32916 },
            { -0.289091, 1.58048, -1.08139, 0.258456, -1.11749 }
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
            { 0.179554, -0.504097 },
            { -0.524464, 0.0395376 },
            { 0.794196, 0.678125 },
            { -0.432833, 1.02502 },
            { -0.67292, -0.228904 } 
        },
        { 
            { -0.190791, 0.271716 },
            { 0.143202, -0.337238 },
            { -0.542431, 0.436789 },
            { 1.0457, 0.336588 },
            { -0.486509, -0.620736 } 
        }
    };
    mat_t AB = BMatrixXdR::fromArray(dataAB).cast<Scalar>().template block<5, 2, 2>(0, 0, 0);

    double determinantData[2][1][1]{
        {{31.1144}},
        {{-43.7003}}
    };
    mat_t determinant = BMatrixXdR::fromArray(determinantData).cast<Scalar>().template block<1, 1, 2>(0, 0, 0);

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
    auto determinantResult = decomposition.determinant().eval();
    assertMatrixEquality(determinant, determinantResult, 1e-3);

    //Test inverse
    auto inverseResult = decomposition.inverse().eval();
    INFO("inverse: \n" << inverseResult);
    assertMatrixEquality(mat_t::Identity(5, 5, 2), A*inverseResult, 1e-5);
}
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
TEST_CASE("LU-Decomposition", "[Dense]")
{
    SECTION("float")
    {
        SECTION("row major")
        {
            testLUDecompositionReal<float, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecompositionReal<float, ColumnMajor>();
        }
    }
    SECTION("double")
    {
        SECTION("row major")
        {
            testLUDecompositionReal<double, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecompositionReal<double, ColumnMajor>();
        }
    }
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
}


template<typename Scalar, int Flags>
void testLUDecompositionLogDet()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, 2, Flags> mat_t;
    double dataA[2][5][5]{
        { 
            { 5.49558, -5.00076, -1.36761, -3.65355, -1.7765 },
            { -5.00076, 7.09413, 0.367989, 2.57289, 3.32289 },
            { -1.36761, 0.367989, 4.45802, 1.96195, -0.529342 },
            { -3.65355, 2.57289, 1.96195, 5.24609, -0.620785 },
            { -1.7765, 3.32289, -0.529342, -0.620785, 9.83736 } 
        },
        { 
            { 8.53528, -0.228827, 1.98238, 5.73122, 2.33181 },
            { -0.228827, 5.163, -1.95338, 1.56358, 3.02351 },
            { 1.98238, -1.95338, 7.18909, -0.663429, -1.657 },
            { 5.73122, 1.56358, -0.663429, 7.75456, 2.6002 },
            { 2.33181, 3.02351, -1.657, 2.6002, 5.06648 } 
        }
    };
    mat_t A = BMatrixXdR::fromArray(dataA).cast<Scalar>().template block<5, 5, 2>(0, 0, 0);
    double logDeterminantData[2][1][1]{
        { { 6.875344150773254} },
        { { 7.55471060202439 } }
    };
    mat_t logDeterminant = BMatrixXdR::fromArray(logDeterminantData).cast<Scalar>().template block<1, 1, 2>(0, 0, 0);

    //perform LU decomposition
    LUDecomposition<mat_t> decomposition(A);
    assertMatrixEquality(logDeterminant, decomposition.logDeterminant(), 1e-3);
}

TEST_CASE("LU-Decomposition LogDet", "[Dense]")
{
    SECTION("float")
    {
        SECTION("row major")
        {
            testLUDecompositionLogDet<float, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecompositionLogDet<float, ColumnMajor>();
        }
    }
    SECTION("double")
    {
        SECTION("row major")
        {
            testLUDecompositionLogDet<double, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecompositionLogDet<double, ColumnMajor>();
        }
    }
}