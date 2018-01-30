#include <catch/catch.hpp>

#include <cuMat/Dense>
#include "Utils.h"

using namespace cuMat;

template<typename Scalar, int Flags>
void testLUDecomposition()
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
TEST_CASE("LU-Decomposition", "[Dense]")
{
    SECTION("float")
    {
        SECTION("row major")
        {
            testLUDecomposition<float, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecomposition<float, ColumnMajor>();
        }
    }
    SECTION("double")
    {
        SECTION("row major")
        {
            testLUDecomposition<float, RowMajor>();
        }
        SECTION("column major")
        {
            testLUDecomposition<float, ColumnMajor>();
        }
    }
}