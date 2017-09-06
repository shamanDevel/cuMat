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

namespace
{
	template <typename M>
	__global__ void TestKernel(M matrix)
	{
		void* mem = matrix.data();
		printf("test, matrix: rows=%d, cols=%d, batches=%d, data=%p\n",
			(int)matrix.rows(), (int)matrix.cols(), (int)matrix.batches(), mem);
		for (Index i = 0; i<0; ++i)
		{
			matrix.rawCoeff(i) = 1;
		}
	}
}

TEST_CASE("ConstantOp", "[nullary]")
{
	typedef cuMat::Matrix<float, 5, 8, 1, cuMat::RowMajor> matrix_t;
	auto setTwoExpr = matrix_t::Constant(5, 8, 1, 2);

	//1. call eval
	matrix_t m1 = setTwoExpr.eval();
	auto m1_host = m1.toEigen();
	REQUIRE(m1_host.isConstant(2));

	//2. cast to matrix
	matrix_t m2 = setTwoExpr;
	auto m2_host = m2.toEigen();
	REQUIRE(m2_host.isConstant(2));
}