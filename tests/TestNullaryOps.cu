#include <catch/catch.hpp>

#include <cuMat/src/Matrix.h>
#include <cuMat/src/EigenInteropHelpers.h>

TEST_CASE("setZero", "[nullary]")
{
	cuMat::Matrix<float, 5, 8, 1, cuMat::RowMajor> m;
	m.setZero();
	auto me = m.toEigen();
	REQUIRE(me.isZero());
}


TEST_CASE("ConstantOp1", "[nullary]")
{
	typedef cuMat::Matrix<float, 5, 8, 1, cuMat::RowMajor> matrix_t;
	auto setTwoExpr = matrix_t::Constant(5, 8, 1, 2);

	//1. call eval
	matrix_t m1 = setTwoExpr.eval();
	auto m1_host = m1.toEigen();
	REQUIRE(m1_host.isConstant(2));
	REQUIRE(m1.rows() == 5);
	REQUIRE(m1.cols() == 8);
	REQUIRE(m1.batches() == 1);

	//2. cast to matrix
	matrix_t m2 = setTwoExpr;
	auto m2_host = m2.toEigen();
	REQUIRE(m2_host.isConstant(2));
}
TEST_CASE("ConstantOp2", "[nullary]")
{
	typedef cuMat::Matrix<float, 5, 8, 1, cuMat::RowMajor> matrix_t;
	auto setTwoExpr = matrix_t::Constant(5);

	//1. call eval
	matrix_t m1 = setTwoExpr.eval();
	auto m1_host = m1.toEigen();
	REQUIRE(m1_host.isConstant(5));
	REQUIRE(m1.rows() == 5);
	REQUIRE(m1.cols() == 8);
	REQUIRE(m1.batches() == 1);

	//2. cast to matrix
	matrix_t m2 = setTwoExpr;
	auto m2_host = m2.toEigen();
	REQUIRE(m2_host.isConstant(5));

	//3. assign to matrix
	matrix_t m3(5, 8, 1);
	m3 = setTwoExpr;
	auto m3_host = m3.toEigen();
	REQUIRE(m3_host.isConstant(5));
}
TEST_CASE("ConstantOp3", "[nullary]")
{
	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor> matrix_t;
	auto setTwoExpr = matrix_t::Constant(20, 30, 5);

	//1. call eval
	matrix_t m1 = setTwoExpr.eval();
	auto m1_host = m1.toEigen();
	REQUIRE(m1_host.isConstant(5));
	REQUIRE(m1.rows() == 20);
	REQUIRE(m1.cols() == 30);
	REQUIRE(m1.batches() == 1);

	//2. cast to matrix
	matrix_t m2 = setTwoExpr;
	auto m2_host = m2.toEigen();
	REQUIRE(m2_host.isConstant(5));
}
TEST_CASE("ConstantOp4", "[nullary]")
{
	typedef cuMat::Matrix<float, cuMat::Dynamic, 8, 1, cuMat::RowMajor> matrix_t;
	auto setTwoExpr = matrix_t::Constant(10, 5);

	//1. call eval
	matrix_t m1 = setTwoExpr.eval();
	auto m1_host = m1.toEigen();
	REQUIRE(m1_host.isConstant(5));
	REQUIRE(m1.rows() == 10);
	REQUIRE(m1.cols() == 8);
	REQUIRE(m1.batches() == 1);

	//2. cast to matrix
	matrix_t m2 = setTwoExpr;
	auto m2_host = m2.toEigen();
	REQUIRE(m2_host.isConstant(5));
}
TEST_CASE("ConstantOp5", "[nullary]")
{
	typedef cuMat::Matrix<float, 3, cuMat::Dynamic, 1, cuMat::RowMajor> matrix_t;
	auto setTwoExpr = matrix_t::Constant(10, 5);

	//1. call eval
	matrix_t m1 = setTwoExpr.eval();
	auto m1_host = m1.toEigen();
	REQUIRE(m1_host.isConstant(5));
	REQUIRE(m1.rows() == 3);
	REQUIRE(m1.cols() == 10);
	REQUIRE(m1.batches() == 1);

	//2. cast to matrix
	matrix_t m2 = setTwoExpr;
	auto m2_host = m2.toEigen();
	REQUIRE(m2_host.isConstant(5));
}