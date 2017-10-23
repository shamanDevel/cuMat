#include <catch/catch.hpp>

#include <cuMat/Core>

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
	//test
    math out21 = mdt2.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out22 = mdt2.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE(in1.transpose() == out21);
    REQUIRE(in2.transpose() == out22);
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
    //test
    math out11 = mdt1.block<6, 5, 1>(0, 0, 0).eval().toEigen();
    math out12 = mdt1.block<6, 5, 1>(0, 0, 1).eval().toEigen();
    REQUIRE((-in1).transpose() == out11);
    REQUIRE((-in2).transpose() == out12);
}