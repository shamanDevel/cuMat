#include <catch/catch.hpp>

#include <cuMat/src/DisableCompilerWarnings.h>
#include <cuMat/src/Matrix.h>
#include <cuMat/src/EigenInteropHelpers.h>

template<typename MatrixType>
__global__ void TestEigenWriteCoeffKernel(dim3 virtual_size, MatrixType matrix)
{
	CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size)
		matrix.coeff(i, j, k, -1) = i + j * 100 + k * 100 * 100;
	CUMAT_KERNEL_3D_LOOP_END
}

template<typename _Matrix>
void testMatrixToEigen(const _Matrix& m)
{
	cuMat::Context& ctx = cuMat::Context::current();
	int sx = m.rows();
	int sy = m.cols();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(sx, sy, 1, TestEigenWriteCoeffKernel<_Matrix>);
	TestEigenWriteCoeffKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, m);
	CUMAT_CHECK_ERROR();

	auto host = m.toEigen();
	for (int y = 0; y<sy; ++y)
	{
		for (int x = 0; x<sx; ++x)
		{
			REQUIRE(host(x, y) == x + y * 100);
		}
	}
}
TEST_CASE("matrix_to_eigen", "[eigen-interop]")
{
    testMatrixToEigen(cuMat::Matrix<float, 1, 1, 1, cuMat::ColumnMajor>(1, 1, 1));
	testMatrixToEigen(cuMat::Matrix<float, 4, 8, 1, cuMat::ColumnMajor>(4, 8, 1));
	testMatrixToEigen(cuMat::Matrix<int, 16, 8, 1, cuMat::ColumnMajor>(16, 8, 1));
	testMatrixToEigen(cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::ColumnMajor>(32, 6, 1));

    testMatrixToEigen(cuMat::Matrix<float, 1, 1, 1, cuMat::RowMajor>(1, 1, 1));
	testMatrixToEigen(cuMat::Matrix<float, 4, 8, 1, cuMat::RowMajor>(4, 8, 1));
	testMatrixToEigen(cuMat::Matrix<int, 16, 8, 1, cuMat::RowMajor>(16, 8, 1));
	testMatrixToEigen(cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::RowMajor>(32, 6, 1));
}

template<typename MatrixType>
__global__ void TestEigenReadCoeffKernel(dim3 virtual_size, MatrixType matrix, int* failure)
{
	CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size)
		if (matrix.coeff(i, j, k, -1) != i + j * 100 + k * 100 * 100)
			failure[0] = 1;
	CUMAT_KERNEL_3D_LOOP_END
}
template <typename _Matrix>
void testMatrixFromEigen(const _Matrix& m)
{
	int sx = m.rows();
	int sy = m.cols();
	_Matrix host = m;

	for (int y = 0; y<sy; ++y)
	{
		for (int x = 0; x<sx; ++x)
		{
			host(x, y) = x + y * 100;
		}
	}

	cuMat::Context& ctx = cuMat::Context::current();

	typedef typename cuMat::eigen::MatrixEigenToCuMat<_Matrix>::type matrix_t;
	matrix_t mat = matrix_t::fromEigen(host);

	cuMat::DevicePointer<int> successFlag(1);
	CUMAT_SAFE_CALL(cudaMemset(successFlag.pointer(), 0, sizeof(int)));

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(sx, sy, 1, TestEigenReadCoeffKernel<matrix_t>);
	TestEigenReadCoeffKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, mat, successFlag.pointer());
	CUMAT_CHECK_ERROR();

	int successFlagHost;
	cudaMemcpy(&successFlagHost, successFlag.pointer(), sizeof(int), cudaMemcpyDeviceToHost);
	REQUIRE(successFlagHost == 0);
}
TEST_CASE("matrix_from_eigen", "[eigen-interop]")
{
	testMatrixFromEigen(Eigen::Matrix<float, 8, 6, Eigen::RowMajor>());
	{
		auto m = Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor>();
		m.resize(12, 6);
		testMatrixFromEigen(m);
	}
	{
		auto m = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();
		m.resize(12, 24);
		testMatrixFromEigen(m);
	}

	testMatrixFromEigen(Eigen::Matrix<float, 8, 6, Eigen::ColMajor>());
	{
		auto m = Eigen::Matrix<float, 16, Eigen::Dynamic, Eigen::ColMajor>();
		m.resize(16, 8);
		testMatrixFromEigen(m);
	}
	{
		auto m = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>();
		m.resize(12, 24);
		testMatrixFromEigen(m);
	}
}
