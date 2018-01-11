#include <catch/catch.hpp>

#include <cuMat/Core>
#include "Utils.h"

using namespace cuMat;

template<typename Scalar>
void testMatrixMatrixDynamic()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, Dynamic, RowMajor> matr;
    typedef Matrix<Scalar, Dynamic, Dynamic, Dynamic, ColumnMajor> matc;

    Scalar dataA[1][2][4] {
        {
            {1, 4, 6, -3},
            {-6, 8, 0, -2}
        }
    };
    Scalar dataB[1][4][3] {
        {
            {-2, 1, 0},
            {5, 7, -3},
            {9, 6, 4},
            {7, -2, -5}
        }
    };
    Scalar dataC[1][2][3] { //C=A*B
        {
            {51, 71, 27},
            {38, 54, -14}
        }
    };

    matr Ar = matr::fromArray(dataA);
    matr Br = matr::fromArray(dataB);
    matr Cr = matr::fromArray(dataC);

    matc Ac = Ar.block(0, 0, 0, 2, 4, 1);
    matc Bc = Br.block(0, 0, 0, 4, 3, 1);
    matr M11 = Ar * Br;
    assertMatrixEquality(Cr, M11);
    matc M12 = Ar * Br;
    assertMatrixEquality(Cr, M12);
    matr M13 = Ac * Br;
    assertMatrixEquality(Cr, M13);
    matc M14 = Ac * Br;
    assertMatrixEquality(Cr, M14);
    matr M15 = Ar * Bc;
    assertMatrixEquality(Cr, M15);
    matc M16 = Ar * Bc;
    assertMatrixEquality(Cr, M16);
    matr M17 = Ac * Bc;
    assertMatrixEquality(Cr, M17);
    matc M18 = Ac * Bc;
    assertMatrixEquality(Cr, M18);
}
TEST_CASE("matrix-matrix dynamic", "[matmul]")
{
    SECTION("float") {
        testMatrixMatrixDynamic<float>();
    }
    SECTION("double") {
        testMatrixMatrixDynamic<double>();
    }
}