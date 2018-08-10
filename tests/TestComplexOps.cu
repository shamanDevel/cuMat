#include <catch/catch.hpp>

#include <cuMat/Core>

#include "Utils.h"

using namespace cuMat;

template<typename type>
void testUnary()
{
    typedef typename internal::NumTraits<type>::RealType real;
    typedef Matrix<type, Dynamic, Dynamic, 1, RowMajor> CMatrix;
    typedef Matrix<real, Dynamic, Dynamic, 1, RowMajor> RMatrix;

    type data[1][3][3]{
        {
            {type(1,0), type(6,0), type(-4,0)}, //only real
            {type(0,5), type(0,-3), type(0, 0.3f)}, //only imaginary
            {type(0.4f,0.9f), type(-1.5f,0.3f), type(3.5f,-2.8f)} //mixed
        }
    };
    CMatrix mat = CMatrix::fromArray(data);

    SECTION("negate")
    {
        type expected[1][3][3]{
            {
                { type(-1,0), type(-6,0), type(4,0) }, //only real
                { type(0,-5), type(0,3), type(0, -0.3f) }, //only imaginary
                { type(-0.4f,-0.9f), type(1.5f,-0.3f), type(-3.5f,2.8f) } //mixed
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), -mat);
    }
    SECTION("abs")
    {
        real expected[1][3][3]{
            {
                { 1, 6, 4 }, //only real
                { 5, 3, 0.3f }, //only imaginary
                { 0.9848858f, 1.529705854f, 4.4821869662f } //mixed
            }
        };
        assertMatrixEquality(RMatrix::fromArray(expected), mat.cwiseAbs(), 1e-5);
    }
    SECTION("abs2")
    {
        real expected[1][3][3]{
            {
                { 1, 36, 16 }, //only real
                { 25, 9, 0.09f }, //only imaginary
                { 0.97f, 2.34f, 20.09f } //mixed
            }
        };
        assertMatrixEquality(RMatrix::fromArray(expected), mat.cwiseAbs2(), 1e-5);
    }
    SECTION("inverse")
    {
        type expected[1][3][3]{
            {
                { type(1, 0), type(1.0f/6, 0), type(1.0f / -4, 0) }, //only real
                { type(0, -1.0f / 5), type(0, 1.0f / 3), type(0, -1.0f / 0.3f) }, //only imaginary
                { type(0.41237113402f, -0.927835051546f), type(-0.641025641f, -0.1282051282f), type(0.17421602787f, 0.1393728223f) } //mixed
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), mat.cwiseInverse(), 1e-5);
    }
    SECTION("conjugate")
    {
        type expected[1][3][3]{
            {
                { type(1,0), type(6,0), type(-4,0) }, //only real
                { type(0,-5), type(0,3), type(0, -0.3f) }, //only imaginary
                { type(0.4f,-0.9f), type(-1.5f,-0.3f), type(3.5f,2.8f) } //mixed
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), mat.conjugate());
    }
}
TEST_CASE("unary", "[complex]")
{
    SECTION("complex-float") {
        testUnary<cfloat>();
    }
    SECTION("complex-double")
    {
        testUnary<cdouble>();
    }
}

template<typename type>
void testBinary()
{
    typedef typename internal::NumTraits<type>::RealType real;
    typedef Matrix<type, 1, Dynamic, 1, RowMajor> CMatrix;
    typedef Matrix<real, 1, Dynamic, 1, RowMajor> RMatrix;

    type dataLeft[1][1][3]{
        {
            { type(0.4f,0.9f), type(-1.5f,0.3f), type(3.5f,-2.8f) }
        }
    };
    CMatrix left = CMatrix::fromArray(dataLeft);

    type dataRight[1][1][3]{
        {
            { type(-3.5f,-0.4f), type(0.1f,2.6f), type(-1.3f,-1.1f) }
        }
    };
    CMatrix right = CMatrix::fromArray(dataRight);

    SECTION("add")
    {
        type expected[1][1][3]{
            {
                { type(-3.1f,0.5f), type(-1.4f,2.9f), type(2.2f,-3.9f) }
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), left + right, 1e-5);
    }
    SECTION("sub")
    {
        type expected[1][1][3]{
            {
                { type(3.9f, 1.3f), type(-1.6f, -2.3f), type(4.8f,-1.7f) }
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), left - right, 1e-5);
    }
    SECTION("mul")
    {
        type expected[1][1][3]{
            {
                { type(-1.04f, -3.31f), type(-0.93f, -3.87f), type(-7.63f, -0.21f) }
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), left.cwiseMul(right), 1e-5);
    }
    SECTION("div")
    {
        type expected[1][1][3]{
            {
                { type(-0.141821f, -0.240935f), type(0.0930576f, 0.580502f), type(-0.506897f, 2.58276f) }
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), left.cwiseDiv(right), 1e-5);
    }
    SECTION("pow")
    {
        type expected[1][1][3]{
            {
                { type(-1.05748f, 1.29578f), type(0.0000841987f, 0.000487057f), type(0.0484779f, -0.0472866f) }
            }
        };
        assertMatrixEquality(CMatrix::fromArray(expected), left.cwisePow(right), 1e-5);
    }
}
TEST_CASE("binary", "[complex]")
{
    SECTION("complex-float") {
        testBinary<cfloat>();
    }
    SECTION("complex-double")
    {
        testBinary<cdouble>();
    }
}


template<typename Type>
void testRealImag()
{
    typedef typename internal::NumTraits<Type>::RealType Real;
    typedef Matrix<Type, Dynamic, Dynamic, 1, RowMajor> CMatrix;
    typedef Matrix<Real, Dynamic, Dynamic, 1, RowMajor> RMatrix;

    Type data[1][3][3]{
        {
            { Type(1,0), Type(6,0), Type(-4,0) }, //only real
            { Type(0,5), Type(0,-3), Type(0, 0.3f) }, //only imaginary
            { Type(0.4f,0.9f), Type(-1.5f,0.3f), Type(3.5f,-2.8f) } //mixed
        }
    };
    CMatrix mat = CMatrix::fromArray(data);

    SECTION("rvalue-direct")
    {
        Real expectedReal[1][3][3]{
            {
                {1, 6, -4},
                {0, 0, 0},
                {0.4f, -1.5f, 3.5f}
            }
        };
        Real expectedImag[1][3][3]{
            {
                { 0, 0, 0 },
                { 5, -3, 0.3f },
                { 0.9f, 0.3f, -2.8f }
            }
        };
        assertMatrixEquality(expectedReal, mat.real());
        assertMatrixEquality(expectedReal, cuMat::functions::real(mat));
        assertMatrixEquality(expectedImag, mat.imag());
        assertMatrixEquality(expectedImag, cuMat::functions::imag(mat));

        RMatrix matr = mat.real(); 
        assertMatrixEquality(expectedReal, matr.real()); //real() of a real matrix is a no-op
    }

    SECTION("lvalue-direct")
    {
        Real realPart[1][3][3]{
            {
                { 1, 6, -4 },
                { 0, 0, 0 },
                { 0.4f, -1.5f, 3.5f }
            }
        };
        Real imagPart[1][3][3]{
            {
                { 0, 0, 0 },
                { 5, -3, 0.3f },
                { 0.9f, 0.3f, -2.8f }
            }
        };
        CMatrix newMat(3, 3);
        newMat.setZero();
        newMat.real() = RMatrix::fromArray(realPart);
        newMat.imag() = RMatrix::fromArray(imagPart);
        assertMatrixEquality(newMat, mat);
    }

    SECTION("rvalue-cwise")
    {
        Real expectedReal[1][3][3]{
            {
                { 1, 6, -4 },
                { 0, 0, 0 },
                { 0.4f, -1.5f, 3.5f }
            }
        };
        Real expectedImag[1][3][3]{
            {
                { 0, 0, 0 },
                { 5, -3, 0.3f },
                { 0.9f, 0.3f, -2.8f }
            }
        };
        auto matExpr = (mat + 0);
        assertMatrixEquality(expectedReal, matExpr.real());
        assertMatrixEquality(expectedReal, cuMat::functions::real(matExpr));
        assertMatrixEquality(expectedImag, matExpr.imag());
        assertMatrixEquality(expectedImag, cuMat::functions::imag(matExpr));
    }
}
TEST_CASE("real+imag", "[complex]")
{
    SECTION("complex-float") {
        testRealImag<cfloat>();
    }
    SECTION("complex-double")
    {
        testRealImag<cdouble>();
    }
}