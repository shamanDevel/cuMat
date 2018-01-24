#include <catch/catch.hpp>

#include <cuMat/Core>
#include "Utils.h"

using namespace cuMat;

TEST_CASE("direct_fixed", "[transpose]")
{
	//direct transposing, fixed matrix
	typedef cuMat::Matrix<float, 5, 6, 2, cuMat::RowMajor> matd2;
	typedef cuMat::Matrix<float, 5, 6, 1, cuMat::RowMajor> matd1;
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> math;
	math in1 = math::Random(5, 6);
	math in2 = math::Random(5, 6);
	matd2 md(5, 6, 2);
	md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
	md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);
    typedef cuMat::Matrix<float, 6, 5, 2, cuMat::RowMajor> matd2t;

    //transpose (for real)
    CUMAT_PROFILING_RESET();
    matd2t mdt2 = md.transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
    REQUIRE(CUMAT_PROFILING_GET(EvalTranspose) == 1);
    assertMatrixEquality(mdt2, md.adjoint());
	//test
    math out21 = mdt2.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out22 = mdt2.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1.transpose() == out21);
    REQUIRE(in2.transpose() == out22);

    //double-transpose
    CUMAT_PROFILING_RESET();
    matd2 md3 = md.transpose().transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 0);
    REQUIRE(CUMAT_PROFILING_GET(EvalTranspose) == 0);
    //test
    math out31 = md3.block<5, 6, 1>(0, 0, 0).eval().toEigen();
    math out32 = md3.block<5, 6, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1 == out31);
    REQUIRE(in2 == out32);
}

TEST_CASE("direct_dynamic1", "[transpose]")
{
    //direct transposing, fixed matrix
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> matd2;
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> matd1;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> math;
    math in1 = math::Random(5, 6);
    math in2 = math::Random(5, 6);
    matd2 md(5, 6, 2);
    md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
    md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);

    //transpose (for real)
    CUMAT_PROFILING_RESET();
    matd2 mdt2 = md.transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
    REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
    assertMatrixEquality(mdt2, md.adjoint());
    //test
    math out21 = mdt2.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out22 = mdt2.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1.transpose() == out21);
    REQUIRE(in2.transpose() == out22);
}
TEST_CASE("direct_dynamic2", "[transpose]")
{
    //direct transposing, fixed matrix
    typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> matd2;
    typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> matd1;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> math;
    math in1 = math::Random(5, 6);
    math in2 = math::Random(5, 6);
    matd2 md(5, 6, 2);
    md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
    md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);

    //transpose (for real)
    CUMAT_PROFILING_RESET();
    matd2 mdt2 = md.transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
    REQUIRE(CUMAT_PROFILING_GET(EvalTranspose) == 1);
    assertMatrixEquality(mdt2, md.adjoint());
    //test
    math out21 = mdt2.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out22 = mdt2.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1.transpose() == out21);
    REQUIRE(in2.transpose() == out22);
}


TEST_CASE("noop_fixed", "[transpose]")
{
    //just changing storage order, fixed matrix
    typedef cuMat::Matrix<int, 5, 6, 2, cuMat::RowMajor> matd2;
    typedef cuMat::Matrix<int, 5, 6, 1, cuMat::RowMajor> matd1;
    typedef Eigen::Matrix<int, 5, 6, Eigen::RowMajor> math;
    math in1 = math::Random(5, 6);
    math in2 = math::Random(5, 6);
    matd2 md(5, 6, 2);
    md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
    md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);
    //transpose (no-op)
    typedef cuMat::Matrix<int, 6, 5, 2, cuMat::ColumnMajor> matdt2;
    typedef Eigen::Matrix<int, 6, 5, Eigen::ColMajor> matht;

    CUMAT_PROFILING_RESET();
    matdt2 mdt1 = md.transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 0);
    assertMatrixEquality(mdt1, md.adjoint());
    //test
    matht out11 = mdt1.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    matht out12 = mdt1.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1.transpose() == out11);
    REQUIRE(in2.transpose() == out12);
}

TEST_CASE("noop_dynamic", "[transpose]")
{
    //just changing storage order, dynamic matrix
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> matd2;
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> matd1;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> math;
    math in1 = math::Random(5, 6);
    math in2 = math::Random(5, 6);
    matd2 md(5, 6, 2);
    md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
    md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);
    //transpose (no-op)
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 2, cuMat::ColumnMajor> matdt2;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> matht;

    CUMAT_PROFILING_RESET();
    matdt2 mdt1 = md.transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 0);
    assertMatrixEquality(mdt1, md.adjoint());
    //test
    math out11 = mdt1.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out12 = mdt1.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1.transpose() == out11);
    REQUIRE(in2.transpose() == out12);
}


TEST_CASE("cwise_fixed", "[transpose]")
{
    //just changing storage order, fixed matrix
    typedef cuMat::Matrix<int, 5, 6, 2, cuMat::RowMajor> matd2;
    typedef cuMat::Matrix<int, 5, 6, 1, cuMat::RowMajor> matd1;
    typedef Eigen::Matrix<int, 5, 6, Eigen::RowMajor> math;
    math in1 = math::Random(5, 6);
    math in2 = math::Random(5, 6);
    matd2 md(5, 6, 2);
    md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
    md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);
    //transpose (no-op)
    typedef cuMat::Matrix<int, 6, 5, 2, cuMat::ColumnMajor> matdt2;
    typedef Eigen::Matrix<int, 6, 5, Eigen::ColMajor> matht;

    CUMAT_PROFILING_RESET();
    matdt2 mdt1 = md.cwiseNegate().transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
    REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
    assertMatrixEquality(mdt1, md.cwiseNegate().adjoint());
    //test
    matht out11 = mdt1.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    matht out12 = mdt1.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE((-in1).transpose() == out11);
    REQUIRE((-in2).transpose() == out12);
}

TEST_CASE("cwise_dynamic", "[transpose]")
{
    //just changing storage order, dynamic matrix
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> matd2;
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> matd1;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> math;
    math in1 = math::Random(5, 6);
    math in2 = math::Random(5, 6);
    matd2 md(5, 6, 2);
    md.block<5, 6, 1>(0, 0, 0) = matd1::fromEigen(in1);
    md.block<5, 6, 1>(0, 0, 1) = matd1::fromEigen(in2);
    //transpose (no-op)
    typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 2, cuMat::ColumnMajor> matdt2;

    CUMAT_PROFILING_RESET();
    matdt2 mdt1 = md.cwiseNegate().transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
    REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
    assertMatrixEquality(mdt1, md.cwiseNegate().adjoint());
    //test
    math out11 = mdt1.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out12 = mdt1.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE((-in1).transpose() == out11);
    REQUIRE((-in2).transpose() == out12);

    //double transpose
    CUMAT_PROFILING_RESET();
    matd2 md3 = md.cwiseNegate().transpose().transpose();
    REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
    REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
    //test
    math out31 = md3.block<5, 6, 1>(0, 0, 0).eval().toEigen();
    math out32 = md3.block<5, 6, 1>(0, 0, 1).eval().toEigen();
    REQUIRE((-in1) == out31);
    REQUIRE((-in2) == out32);
}

template<typename Type>
void testComplexTranspose()
{
    typedef typename internal::NumTraits<Type>::RealType Real;
    typedef Matrix<Type, Dynamic, Dynamic, 1, RowMajor> CMatrix;
    typedef Matrix<Type, Dynamic, Dynamic, 1, ColumnMajor> CMatrixT;
    typedef Matrix<Real, Dynamic, Dynamic, 1, RowMajor> RMatrix;

    Type data[1][3][3]{
        {
            { Type(1,0), Type(6,0), Type(-4,0) }, //only real
            { Type(0,5), Type(0,-3), Type(0, 0.3f) }, //only imaginary
            { Type(0.4f,0.9f), Type(-1.5f,0.3f), Type(3.5f,-2.8f) } //mixed
        }
    };
    CMatrix mat = CMatrix::fromArray(data);

    Type transposedData[1][3][3]{
        {
            { Type(1,0), Type(0,5), Type(0.4f,0.9f) },
            { Type(6,0), Type(0,-3), Type(-1.5f,0.3f) },
            { Type(-4,0), Type(0, 0.3f), Type(3.5f,-2.8f) }
        }
    };
    CMatrix transposed = CMatrix::fromArray(transposedData);

    Type adjointData[1][3][3]{
        {
            { Type(1,0), Type(0,-5), Type(0.4f,-0.9f) },
            { Type(6,0), Type(0,3), Type(-1.5f,-0.3f) },
            { Type(-4,0), Type(0, -0.3f), Type(3.5f,2.8f) }
        }
    };
    CMatrix adjoint = CMatrix::fromArray(adjointData);

    SECTION("transpose")
    {
        //no-op
        CUMAT_PROFILING_RESET();
        CMatrixT m1 = mat.transpose();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 0);
        assertMatrixEquality(transposed, m1);

        //direct
        CUMAT_PROFILING_RESET();
        CMatrix m2 = mat.transpose();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalTranspose) == 1);
        assertMatrixEquality(transposed, m2);

        //cwise 1
        CUMAT_PROFILING_RESET();
        CMatrixT m3 = (mat + 0).transpose();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
        assertMatrixEquality(transposed, m3);

        //cwise 1
        CUMAT_PROFILING_RESET();
        CMatrix m4 = (mat + 0).transpose();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
        assertMatrixEquality(transposed, m4);
    }

    SECTION("adjoint")
    {
        //no-op
        CUMAT_PROFILING_RESET();
        CMatrixT m1 = mat.adjoint();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
        assertMatrixEquality(adjoint, m1);

        //direct
        CUMAT_PROFILING_RESET();
        CMatrix m2 = mat.adjoint();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalTranspose) == 1);
        assertMatrixEquality(adjoint, m2);

        //cwise 1
        CUMAT_PROFILING_RESET();
        CMatrixT m3 = (mat + 0).adjoint();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
        assertMatrixEquality(adjoint, m3);

        //cwise 1
        CUMAT_PROFILING_RESET();
        CMatrix m4 = (mat + 0).adjoint();
        REQUIRE(CUMAT_PROFILING_GET(EvalAny) == 1);
        REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == 1);
        assertMatrixEquality(adjoint, m4);
    }
}
TEST_CASE("complex", "[transpose]")
{
    SECTION("float")
    {
        testComplexTranspose<cfloat>();
    }
    SECTION("double")
    {
        testComplexTranspose<cdouble>();
    }
}