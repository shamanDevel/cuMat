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

    SECTION("regular")
    {
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

    SECTION("transposed-left")
    {
        Ar = Ar.transpose();
        matc Ac = Ar.block(0, 0, 0, 4, 2, 1);
        matc Bc = Br.block(0, 0, 0, 4, 3, 1);
        matr M11 = Ar.transpose() * Br;
        assertMatrixEquality(Cr, M11);
        matc M12 = Ar.transpose() * Br;
        assertMatrixEquality(Cr, M12);
        matr M13 = Ac.transpose() * Br;
        assertMatrixEquality(Cr, M13);
        matc M14 = Ac.transpose() * Br;
        assertMatrixEquality(Cr, M14);
        matr M15 = Ar.transpose() * Bc;
        assertMatrixEquality(Cr, M15);
        matc M16 = Ar.transpose() * Bc;
        assertMatrixEquality(Cr, M16);
        matr M17 = Ac.transpose() * Bc;
        assertMatrixEquality(Cr, M17);
        matc M18 = Ac.transpose() * Bc;
        assertMatrixEquality(Cr, M18);
    }

    SECTION("transposed-right")
    {
        Br = Br.transpose();
        matc Ac = Ar.block(0, 0, 0, 2, 4, 1);
        matc Bc = Br.block(0, 0, 0, 3, 4, 1);
        matr M11 = Ar * Br.transpose();
        assertMatrixEquality(Cr, M11);
        matc M12 = Ar * Br.transpose();
        assertMatrixEquality(Cr, M12);
        matr M13 = Ac * Br.transpose();
        assertMatrixEquality(Cr, M13);
        matc M14 = Ac * Br.transpose();
        assertMatrixEquality(Cr, M14);
        matr M15 = Ar * Bc.transpose();
        assertMatrixEquality(Cr, M15);
        matc M16 = Ar * Bc.transpose();
        assertMatrixEquality(Cr, M16);
        matr M17 = Ac * Bc.transpose();
        assertMatrixEquality(Cr, M17);
        matc M18 = Ac * Bc.transpose();
        assertMatrixEquality(Cr, M18);
    }

    SECTION("transposed-left + transposed-right")
    {
        Ar = Ar.transpose();
        Br = Br.transpose();
        matc Ac = Ar.block(0, 0, 0, 4, 2, 1);
        matc Bc = Br.block(0, 0, 0, 3, 4, 1);
        matr M11 = Ar.transpose() * Br.transpose();
        assertMatrixEquality(Cr, M11);
        matc M12 = Ar.transpose() * Br.transpose();
        assertMatrixEquality(Cr, M12);
        matr M13 = Ac.transpose() * Br.transpose();
        assertMatrixEquality(Cr, M13);
        matc M14 = Ac.transpose() * Br.transpose();
        assertMatrixEquality(Cr, M14);
        matr M15 = Ar.transpose() * Bc.transpose();
        assertMatrixEquality(Cr, M15);
        matc M16 = Ar.transpose() * Bc.transpose();
        assertMatrixEquality(Cr, M16);
        matr M17 = Ac.transpose() * Bc.transpose();
        assertMatrixEquality(Cr, M17);
        matc M18 = Ac.transpose() * Bc.transpose();
        assertMatrixEquality(Cr, M18);
    }

    SECTION("transposed-output")
    {
        Cr = Cr.transpose();
        matc Ac = Ar.block(0, 0, 0, 2, 4, 1);
        matc Bc = Br.block(0, 0, 0, 4, 3, 1);
        matr M11 = (Ar * Br).transpose();
        assertMatrixEquality(Cr, M11);
        matc M12 = (Ar * Br).transpose();
        assertMatrixEquality(Cr, M12);
        matr M13 = (Ac * Br).transpose();
        assertMatrixEquality(Cr, M13);
        matc M14 = (Ac * Br).transpose();
        assertMatrixEquality(Cr, M14);
        matr M15 = (Ar * Bc).transpose();
        assertMatrixEquality(Cr, M15);
        matc M16 = (Ar * Bc).transpose();
        assertMatrixEquality(Cr, M16);
        matr M17 = (Ac * Bc).transpose();
        assertMatrixEquality(Cr, M17);
        matc M18 = (Ac * Bc).transpose();
        assertMatrixEquality(Cr, M18);
    }

    SECTION("transposed-left + transposed-output")
    {
        Ar = Ar.transpose();
        Cr = Cr.transpose();
        matc Ac = Ar.block(0, 0, 0, 4, 2, 1);
        matc Bc = Br.block(0, 0, 0, 4, 3, 1);
        matr M11 = (Ar.transpose() * Br).transpose();
        assertMatrixEquality(Cr, M11);
        matc M12 = (Ar.transpose() * Br).transpose();
        assertMatrixEquality(Cr, M12);
        matr M13 = (Ac.transpose() * Br).transpose();
        assertMatrixEquality(Cr, M13);
        matc M14 = (Ac.transpose() * Br).transpose();
        assertMatrixEquality(Cr, M14);
        matr M15 = (Ar.transpose() * Bc).transpose();
        assertMatrixEquality(Cr, M15);
        matc M16 = (Ar.transpose() * Bc).transpose();
        assertMatrixEquality(Cr, M16);
        matr M17 = (Ac.transpose() * Bc).transpose();
        assertMatrixEquality(Cr, M17);
        matc M18 = (Ac.transpose() * Bc).transpose();
        assertMatrixEquality(Cr, M18);
    }

    SECTION("transposed-right + transposed-output")
    {
        Br = Br.transpose();
        Cr = Cr.transpose();
        matc Ac = Ar.block(0, 0, 0, 2, 4, 1);
        matc Bc = Br.block(0, 0, 0, 3, 4, 1);
        matr M11 = (Ar * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M11);
        matc M12 = (Ar * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M12);
        matr M13 = (Ac * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M13);
        matc M14 = (Ac * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M14);
        matr M15 = (Ar * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M15);
        matc M16 = (Ar * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M16);
        matr M17 = (Ac * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M17);
        matc M18 = (Ac * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M18);
    }

    SECTION("transposed-left + transposed-right + transposed-output")
    {
        Ar = Ar.transpose();
        Br = Br.transpose();
        Cr = Cr.transpose();
        matc Ac = Ar.block(0, 0, 0, 4, 2, 1);
        matc Bc = Br.block(0, 0, 0, 3, 4, 1);
        matr M11 = (Ar.transpose() * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M11);
        matc M12 = (Ar.transpose() * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M12);
        matr M13 = (Ac.transpose() * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M13);
        matc M14 = (Ac.transpose() * Br.transpose()).transpose();
        assertMatrixEquality(Cr, M14);
        matr M15 = (Ar.transpose() * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M15);
        matc M16 = (Ar.transpose() * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M16);
        matr M17 = (Ac.transpose() * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M17);
        matc M18 = (Ac.transpose() * Bc.transpose()).transpose();
        assertMatrixEquality(Cr, M18);
    }
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


template<typename Scalar>
void testMatrixVector()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, 1, RowMajor> matr;
    typedef Matrix<Scalar, Dynamic, Dynamic, 1, ColumnMajor> matc;
    typedef Matrix<Scalar, Dynamic, 1, 1, RowMajor> vecr;
    typedef Matrix<Scalar, Dynamic, 1, 1, ColumnMajor> vecc;

    Scalar dataA[1][3][4]{
        {
            { 1, 4, 6, -3 },
            { -6, 8, 0, -2 },
            {5, 2, -7, 8}
        }
    };
    Scalar dataB[1][4][1]{
        {
            { -2 },
            { 5 },
            { 9 },
            { 7 }
        }
    };
    Scalar dataC[1][3][1]{ //C=A*B
        {
            { 51 },
            { 38 },
            {-7}
        }
    };

    matr Ar = matr::fromArray(dataA);
    vecr Br = vecr::fromArray(dataB);
    vecr Cr = vecr::fromArray(dataC);

    matc Ac = Ar.block(0, 0, 0, 3, 4, 1);
    vecc Bc = Br.block(0, 0, 0, 4, 1, 1);

    vecr M11 = Ar * Br;
    assertMatrixEquality(Cr, M11);
    vecc M12 = Ar * Br;
    assertMatrixEquality(Cr, M12);
    vecr M13 = Ac * Br;
    assertMatrixEquality(Cr, M13);
    vecc M14 = Ac * Br;
    assertMatrixEquality(Cr, M14);
    vecr M15 = Ar * Bc;
    assertMatrixEquality(Cr, M15);
    vecc M16 = Ar * Bc;
    assertMatrixEquality(Cr, M16);
    vecr M17 = Ac * Bc;
    assertMatrixEquality(Cr, M17);
    vecc M18 = Ac * Bc;
    assertMatrixEquality(Cr, M18);
}
TEST_CASE("matrix-vector", "[matmul]")
{
    SECTION("float") {
        testMatrixVector<float>();
    }
    SECTION("double") {
        testMatrixVector<double>();
    }
}


template<typename Scalar>
void testVectorMatrix()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, 1, RowMajor> matr;
    typedef Matrix<Scalar, Dynamic, Dynamic, 1, ColumnMajor> matc;
    typedef Matrix<Scalar, 1, Dynamic, 1, RowMajor> vecr;
    typedef Matrix<Scalar, 1, Dynamic, 1, ColumnMajor> vecc;

    Scalar dataA[1][1][4]{
        {
            { -2, 5, 9, 7 }
        }
    };
    Scalar dataB[1][4][3]{
        {
            { 1, 4, 6 },
            { -6, 8, 0 },
            { 5, 2, -7 },
            { 3, 0, -1}
        }
    };
    Scalar dataC[1][1][3]{ //C=A*B
        {
            { 34, 50, -82 }
        }
    };

    vecr Ar = vecr::fromArray(dataA);
    matr Br = matr::fromArray(dataB);
    vecr Cr = vecr::fromArray(dataC);

    vecr Ac = Ar.block(0, 0, 0, 1, 4, 1);
    matr Bc = Br.block(0, 0, 0, 4, 3, 1);

    vecr M11 = Ar * Br;
    assertMatrixEquality(Cr, M11);
    vecc M12 = Ar * Br;
    assertMatrixEquality(Cr, M12);
    vecr M13 = Ac * Br;
    assertMatrixEquality(Cr, M13);
    vecc M14 = Ac * Br;
    assertMatrixEquality(Cr, M14);
    vecr M15 = Ar * Bc;
    assertMatrixEquality(Cr, M15);
    vecc M16 = Ar * Bc;
    assertMatrixEquality(Cr, M16);
    vecr M17 = Ac * Bc;
    assertMatrixEquality(Cr, M17);
    vecc M18 = Ac * Bc;
    assertMatrixEquality(Cr, M18);
}
TEST_CASE("vector-matrix", "[matmul]")
{
    SECTION("float") {
        testVectorMatrix<float>();
    }
    SECTION("double") {
        testVectorMatrix<double>();
    }
}



template<typename Scalar>
void testVectorVector()
{
    typedef Matrix<Scalar, Dynamic, Dynamic, 1, RowMajor> matr;
    typedef Matrix<Scalar, Dynamic, Dynamic, 1, ColumnMajor> matc;
    typedef Matrix<Scalar, 1, Dynamic, 1, RowMajor> rvecr;
    typedef Matrix<Scalar, 1, Dynamic, 1, ColumnMajor> rvecc;
    typedef Matrix<Scalar, Dynamic, 1, 1, RowMajor> cvecr;
    typedef Matrix<Scalar, Dynamic, 1, 1, ColumnMajor> cvecc;
    typedef Matrix<Scalar, 1, 1, 1, RowMajor> scalarr;
    typedef Matrix<Scalar, 1, 1, 1, ColumnMajor> scalarc;

    Scalar dataA[1][1][4]{
        {
            { -2, 5, 9, 7 }
        }
    };
    Scalar dataB[1][4][1]{
        {
            { 1 },
            { -6 },
            { 5 },
            { 3 }
        }
    };
    Scalar dataAB[1][1][1]{
        {
            { 34 }
        }
    };
    Scalar dataBA[1][4][4]{
        {
            { -2, 5, 9, 7 },
            { 12, -30, -54, -42 },
            { -10, 25, 45, 35 },
            { -6, 15, 27, 21 }
        }
    };

    rvecr Ar = rvecr::fromArray(dataA);
    cvecr Br = cvecr::fromArray(dataB);
    scalarr ABr = scalarr::fromArray(dataAB);
    matr BAr = matr::fromArray(dataBA);

    rvecc Ac = Ar.block(0, 0, 0, 1, 4, 1);
    cvecc Bc = Br.block(0, 0, 0, 4, 1, 1);

    SECTION("A*B -> scalar") {
        scalarr M11 = Ar * Br;
        assertMatrixEquality(ABr, M11);
        scalarc M12 = Ar * Br;
        assertMatrixEquality(ABr, M12);
        scalarr M13 = Ac * Br;
        assertMatrixEquality(ABr, M13);
        scalarc M14 = Ac * Br;
        assertMatrixEquality(ABr, M14);
        scalarr M15 = Ar * Bc;
        assertMatrixEquality(ABr, M15);
        scalarc M16 = Ar * Bc;
        assertMatrixEquality(ABr, M16);
        scalarr M17 = Ac * Bc;
        assertMatrixEquality(ABr, M17);
        scalarc M18 = Ac * Bc;
        assertMatrixEquality(ABr, M18);
    }

    SECTION("B*A -> matrix") {
        matr M11 = Br * Ar;
        assertMatrixEquality(BAr, M11);
        matc M12 = Br * Ar;
        assertMatrixEquality(BAr, M12);
        matr M13 = Bc * Ar;
        assertMatrixEquality(BAr, M13);
        matc M14 = Bc * Ar;
        assertMatrixEquality(BAr, M14);
        matr M15 = Br * Ac;
        assertMatrixEquality(BAr, M15);
        matc M16 = Br * Ac;
        assertMatrixEquality(BAr, M16);
        matr M17 = Bc * Ac;
        assertMatrixEquality(BAr, M17);
        matc M18 = Bc * Ac;
        assertMatrixEquality(BAr, M18);
    }
}
TEST_CASE("vector-vector", "[matmul]")
{
    SECTION("float") {
        testVectorVector<float>();
    }
    SECTION("double") {
        testVectorVector<double>();
    }
}