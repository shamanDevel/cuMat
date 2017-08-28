#include <catch/catch.hpp>

#include <cuMat/src/Matrix.h>

#define TEST_SIZE_F1(type, flags, rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime) \
	do{ \
	cuMat::Matrix<type, rowCompile, colCompile, batchCompile, flags> m(rowRuntime, colRuntime, batchRuntime); \
	REQUIRE(m.rows() == rowRuntime); \
	REQUIRE(m.cols() == colRuntime); \
	REQUIRE(m.batches() == batchRuntime); \
	REQUIRE(m.size() == rowRuntime*colRuntime*batchRuntime); \
	if (m.size()>0) REQUIRE(m.data() != nullptr); \
	}while(false)

#define TEST_SIZE_F2(rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime) \
	do{ \
	TEST_SIZE_F1(bool, cuMat::RowMajorBit,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(bool, cuMat::ColumnMajorBit,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(int, cuMat::RowMajorBit,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(int, cuMat::ColumnMajorBit,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(float, cuMat::RowMajorBit,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(float, cuMat::ColumnMajorBit,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(double, cuMat::RowMajorBit,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(double, cuMat::ColumnMajorBit,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	}while(false)
	

TEST_CASE("instantiation_fully", "[matrix]")
{
	TEST_SIZE_F2(0, 0, 0, 0, 0, 0);

	TEST_SIZE_F2(1, 1, 1, 1, 1, 1);
	TEST_SIZE_F2(8, 8, 1, 1, 1, 1);
	TEST_SIZE_F2(1, 1, 8, 8, 1, 1);
	TEST_SIZE_F2(1, 1, 1, 1, 8, 8);
	TEST_SIZE_F2(8, 8, 8, 8, 1, 1);
	TEST_SIZE_F2(8, 8, 1, 1, 8, 8);
	TEST_SIZE_F2(1, 1, 8, 8, 8, 8);

	TEST_SIZE_F2(cuMat::Dynamic, 16, 4, 4, 4, 4);
	TEST_SIZE_F2(4, 4, cuMat::Dynamic, 16, 4, 4);
	TEST_SIZE_F2(4, 4, 4, 4, cuMat::Dynamic, 16);
	TEST_SIZE_F2(cuMat::Dynamic, 16, cuMat::Dynamic, 8, 4, 4);
	TEST_SIZE_F2(4, 4, cuMat::Dynamic, 16, cuMat::Dynamic, 8);
	TEST_SIZE_F2(cuMat::Dynamic, 8, 4, 4, cuMat::Dynamic, 16);
	TEST_SIZE_F2(cuMat::Dynamic, 8, cuMat::Dynamic, 32, cuMat::Dynamic, 16);
}

#define TEST_SIZE_D1(type, flags, Rows, Cols, Batches) \
	do { \
	cuMat::Matrix<type, Rows, Cols, Batches, flags> m; \
	if (Rows > 0) {\
		REQUIRE(m.rows() == Rows); \
	} else {\
		REQUIRE(m.rows() == 0); \
	} if (Cols > 0) { \
		REQUIRE(m.cols() == Cols); \
	} else {\
		REQUIRE(m.cols() == 0); \
	} if (Batches > 0) { \
		REQUIRE(m.batches() == Batches); \
	} else {\
		REQUIRE(m.batches() == 0); \
	} if (Rows>0 && Cols>0 && Batches>0) { \
		REQUIRE(m.data() != nullptr); \
	} else {\
		REQUIRE(m.data() == nullptr); \
	} \
	} while (false)
#define TEST_SIZE_D2(rows, cols, batches) \
	do { \
	TEST_SIZE_D1(bool, cuMat::RowMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(bool, cuMat::ColumnMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(int, cuMat::RowMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(int, cuMat::ColumnMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(float, cuMat::RowMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(float, cuMat::ColumnMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(double, cuMat::RowMajorBit, rows, cols, batches); \
	TEST_SIZE_D1(double, cuMat::ColumnMajorBit, rows, cols, batches); \
	} while(false)

TEST_CASE("instantiation_default", "[matrix]")
{
	TEST_SIZE_D2(2, 4, 8);
	TEST_SIZE_D2(cuMat::Dynamic, 4, 8);
	TEST_SIZE_D2(2, cuMat::Dynamic, 8);
	TEST_SIZE_D2(2, 4, cuMat::Dynamic);
	TEST_SIZE_D2(cuMat::Dynamic, cuMat::Dynamic, 8);
	TEST_SIZE_D2(cuMat::Dynamic, 4, cuMat::Dynamic);
	TEST_SIZE_D2(2, cuMat::Dynamic, cuMat::Dynamic);
	TEST_SIZE_D2(cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic);
}

TEST_CASE("instantiation_vector", "[matrix]")
{
	cuMat::Matrix<float, 1, cuMat::Dynamic, 1, 0> columnV(8);
	REQUIRE(columnV.rows() == 1);
	REQUIRE(columnV.cols() == 8);
	REQUIRE(columnV.batches() == 1);
	cuMat::Matrix<float, cuMat::Dynamic, 1, 1, 0> rowV(8);
	REQUIRE(rowV.rows() == 8);
	REQUIRE(rowV.cols() == 1);
	REQUIRE(rowV.batches() == 1);
}

#define TEST_SIZE_M(rowCompile, rowRuntime, colCompile, colRuntime) \
	do {\
	cuMat::Matrix<float, rowCompile, colCompile, 1, 0> m(rowRuntime, colRuntime); \
	REQUIRE(m.rows() == rowRuntime); \
	REQUIRE(m.cols() == colRuntime); \
	REQUIRE(m.batches() == 1); \
	REQUIRE(m.size() == rowRuntime*colRuntime); \
	} while(0)
TEST_CASE("instantiation_matrix", "[matrix]")
{
	TEST_SIZE_M(4, 4, 8, 8);
	TEST_SIZE_M(cuMat::Dynamic, 4, 8, 8);
	TEST_SIZE_M(4, 4, cuMat::Dynamic, 8);
	TEST_SIZE_M(cuMat::Dynamic, 4, cuMat::Dynamic, 8);
}

TEST_CASE("instantiation_throws", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();
	REQUIRE_THROWS((cuMat::Matrix<float, 8, 6, 4, 0>(7, 6, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, 8, 6, 4, 0>(8, 7, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, 8, 6, 4, 0>(8, 6, 3)));

	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, 6, 4, 0>(-1, 6, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, 8, cuMat::Dynamic, 4, 0>(8, -1, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, 8, 6, cuMat::Dynamic, 0>(8, 6, -1)));

	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 4, 0>(-1, 6, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 4, 0>(8, -1, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, 6, cuMat::Dynamic, 0>(-1, 6, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, 6, cuMat::Dynamic, 0>(8, 6, -1)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, 6, cuMat::Dynamic, 0>(8, -1, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, 6, cuMat::Dynamic, 0>(8, 6, -1)));

	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0>(-1, 6, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0>(8, -1, 4)));
	REQUIRE_THROWS((cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0>(8, 6, -1)));
}


template<typename MatrixType>
__global__ void TestMatrixWriteRawKernel(dim3 virtual_size, MatrixType matrix)
{
	CUMAT_KERNEL_1D_LOOP(i, virtual_size)
	{
		matrix.rawCoeff(i) = i;
	}
}
//Tests if a kernel can write the raw data
TEST_CASE("write_raw", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0> m(sx, sy, sz);

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D((unsigned int) m.size());
	TestMatrixWriteRawKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, m);
	CUMAT_CHECK_ERROR();

	std::vector<int> host(sx * sy * sz);
	m.copyToHost(&host[0]);
	for (int i=0; i<sx*sy*sz; ++i)
	{
		REQUIRE(host[i] == i);
	}
}

template<typename MatrixType>
__global__ void TestMatrixReadRawKernel(dim3 virtual_size, MatrixType matrix, int* failure)
{
	CUMAT_KERNEL_1D_LOOP(i, virtual_size)
	{
		if (matrix.rawCoeff(i) != i) failure[0] = 1;
	}
}
//Test if the kernel can read the raw data
TEST_CASE("read_raw", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0> m(sx, sy, sz);

	std::vector<int> host1(sx * sy * sz);
	for (int i = 0; i<sx*sy*sz; ++i)
	{
		host1[i] = i;
	}
	m.copyFromHost(host1.data());

	cuMat::DevicePointer<int> successFlag(1);
	CUMAT_SAFE_CALL(cudaMemset(successFlag.pointer(), 0, sizeof(int)));

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D((unsigned int) m.size());
	TestMatrixReadRawKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, m, successFlag.pointer());
	CUMAT_CHECK_ERROR();

	int successFlagHost;
	cudaMemcpy(&successFlagHost, successFlag.pointer(), sizeof(int), cudaMemcpyDeviceToHost);
	REQUIRE(successFlagHost == 0);
}


template<typename MatrixType>
__global__ void TestMatrixWriteCoeffKernel(dim3 virtual_size, MatrixType matrix)
{
	CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size)
	{
		matrix.coeff(i, j, k) = i + j*100 + k * 100*100;
	}
}
//Tests if a kernel can write the 3d-indexed coefficients
TEST_CASE("write_coeff_columnMajor", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajorBit> m(sx, sy, sz);

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(sx, sy, sz);
	TestMatrixWriteCoeffKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, m);
	CUMAT_CHECK_ERROR();

	std::vector<int> host(sx * sy * sz);
	m.copyToHost(&host[0]);
	int i = 0;
	for (int z=0; z<sz; ++z)
	{
		for (int y=0; y<sy; ++y)
		{
			for (int x=0; x<sx; ++x)
			{
				REQUIRE(host[i] == x + y * 100 + z * 100 * 100);
				i++;
			}
		}
	}
}
//Tests if a kernel can write the 3d-indexed coefficients
TEST_CASE("write_coeff_rowMajor", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajorBit> m(sx, sy, sz);

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(sx, sy, sz);
	TestMatrixWriteCoeffKernel <<< cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
		(cfg.virtual_size, m);
	CUMAT_CHECK_ERROR();

	std::vector<int> host(sx * sy * sz);
	m.copyToHost(&host[0]);
	int i = 0;
	for (int z = 0; z<sz; ++z)
	{
		for (int x = 0; x<sx; ++x)
		{
			for (int y = 0; y<sy; ++y)
			{
				REQUIRE(host[i] == x + y * 100 + z * 100 * 100);
				i++;
			}
		}
	}
}