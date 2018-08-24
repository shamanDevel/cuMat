#include <catch/catch.hpp>

#include <cuMat/Core>
#include "Utils.h"

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
	TEST_SIZE_F1(bool, cuMat::RowMajor,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(bool, cuMat::ColumnMajor,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(int, cuMat::RowMajor,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(int, cuMat::ColumnMajor,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(float, cuMat::RowMajor,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(float, cuMat::ColumnMajor,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(double, cuMat::RowMajor,   rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
	TEST_SIZE_F1(double, cuMat::ColumnMajor,rowCompile, rowRuntime, colCompile, colRuntime, batchCompile, batchRuntime); \
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
	TEST_SIZE_D1(bool, cuMat::RowMajor, rows, cols, batches); \
	TEST_SIZE_D1(bool, cuMat::ColumnMajor, rows, cols, batches); \
	TEST_SIZE_D1(int, cuMat::RowMajor, rows, cols, batches); \
	TEST_SIZE_D1(int, cuMat::ColumnMajor, rows, cols, batches); \
	TEST_SIZE_D1(float, cuMat::RowMajor, rows, cols, batches); \
	TEST_SIZE_D1(float, cuMat::ColumnMajor, rows, cols, batches); \
	TEST_SIZE_D1(double, cuMat::RowMajor, rows, cols, batches); \
	TEST_SIZE_D1(double, cuMat::ColumnMajor, rows, cols, batches); \
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


TEST_CASE("index_computations_rowMajor", "[matrix]")
{
	cuMat::Matrix<int, 5, 16, 7, cuMat::RowMajor> m;
	for (cuMat::Index i=0; i<m.rows(); ++i)
	{
		for (cuMat::Index j=0; j<m.cols(); ++j)
		{
			for (cuMat::Index k=0; k<m.batches(); ++k)
			{
				cuMat::Index index = m.index(i, j, k);
				REQUIRE(index >= 0);
				REQUIRE(index < m.size());
				cuMat::Index i2, j2, k2;
				m.index(index, i2, j2, k2);
				REQUIRE(i2 == i);
				REQUIRE(j2 == j);
				REQUIRE(k2 == k);
			}
		}
	}
}
TEST_CASE("index_computations_columnMajor", "[matrix]")
{
	cuMat::Matrix<int, 5, 16, 7, cuMat::ColumnMajor> m;
	for (cuMat::Index i = 0; i<m.rows(); ++i)
	{
		for (cuMat::Index j = 0; j<m.cols(); ++j)
		{
			for (cuMat::Index k = 0; k<m.batches(); ++k)
			{
				cuMat::Index index = m.index(i, j, k);
				REQUIRE(index >= 0);
				REQUIRE(index < m.size());
				cuMat::Index i2, j2, k2;
				m.index(index, i2, j2, k2);
				REQUIRE(i2 == i);
				REQUIRE(j2 == j);
				REQUIRE(k2 == k);
			}
		}
	}
}

template<typename MatrixType>
__global__ void TestMatrixWriteRawKernel(dim3 virtual_size, MatrixType matrix)
{
	CUMAT_KERNEL_1D_LOOP(i, virtual_size)
		matrix.setRawCoeff(i, i);
	CUMAT_KERNEL_1D_LOOP_END
}
//Tests if a kernel can write the raw data
TEST_CASE("write_raw", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0> Mat_t;
	Mat_t m(sx, sy, sz);

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D((unsigned int) m.size(), TestMatrixWriteRawKernel<Mat_t>);
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
		if (matrix.rawCoeff(i) != i) failure[0] = 1;
	CUMAT_KERNEL_1D_LOOP_END
}
//Test if the kernel can read the raw data
TEST_CASE("read_raw", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, 0> Mat_t;
	Mat_t m(sx, sy, sz);

	std::vector<int> host1(sx * sy * sz);
	for (int i = 0; i<sx*sy*sz; ++i)
	{
		host1[i] = i;
	}
	m.copyFromHost(host1.data());

	cuMat::DevicePointer<int> successFlag(1);
	CUMAT_SAFE_CALL(cudaMemset(successFlag.pointer(), 0, sizeof(int)));

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig1D((unsigned int) m.size(), TestMatrixReadRawKernel<Mat_t>);
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
		matrix.coeff(i, j, k, -1) = i + j*100 + k * 100*100;
	CUMAT_KERNEL_3D_LOOP_END
}
//Tests if a kernel can write the 3d-indexed coefficients
TEST_CASE("write_coeff_columnMajor", "[matrix]")
{
	cuMat::Context& ctx = cuMat::Context::current();

	int sx = 4;
	int sy = 8;
	int sz = 16;
	typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> Mat_t;
	Mat_t m(sx, sy, sz);

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(sx, sy, sz, TestMatrixWriteCoeffKernel<Mat_t>);
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
	typedef cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> Mat_t;
	Mat_t m(sx, sy, sz);

	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(sx, sy, sz, TestMatrixWriteCoeffKernel<Mat_t>);
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


TEST_CASE("from_array", "[matrix]")
{
    int data[2][4][3] = {
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10,11,12}
        },
        {
            {13,14,15},
            {16,17,18},
            {19,20,21},
            {22,23,24}
        }
    };
    cuMat::BMatrixXiR m = cuMat::BMatrixXiR::fromArray(data);
    REQUIRE(m.rows() == 4);
    REQUIRE(m.cols() == 3);
    REQUIRE(m.batches() == 2);
    std::vector<int> mem(24);
    m.copyToHost(&mem[0]);
    int i = 0;
    for (int z = 0; z<m.batches(); ++z)
    {
        for (int x = 0; x<m.rows(); ++x)
        {
            for (int y = 0; y<m.cols(); ++y)
            {
                INFO("x=" << x << ", y=" << y << ", z=" << z);
                REQUIRE(mem[i] == y + x * 3 + z * 3 * 4 + 1);
                i++;
            }
        }
    }
}


// Matrix assignments

TEST_CASE("assign", "[matrix]")
{
	cuMat::Matrix<int, 5, 7, 3, cuMat::RowMajor> mat1;
	REQUIRE(mat1.dataPointer().getCounter() == 1);
	
	cuMat::Matrix<int, cuMat::Dynamic, 7, 3, cuMat::RowMajor> mat2(mat1);
	REQUIRE(mat1.dataPointer().getCounter() == 2);
	
	cuMat::Matrix<int, 5, 7, cuMat::Dynamic, cuMat::RowMajor> mat3;
	mat3 = mat1;
	REQUIRE(mat1.dataPointer().getCounter() == 3);

	cuMat::Matrix<int, cuMat::Dynamic, 7, cuMat::Dynamic, cuMat::RowMajor> mat4(mat3);
	REQUIRE(mat1.dataPointer().getCounter() == 4);
	REQUIRE(mat4.dataPointer().getCounter() == 4);
	
	REQUIRE(mat1.data() == mat2.data());
	REQUIRE(mat1.data() == mat3.data());
	REQUIRE(mat1.data() == mat4.data());
}

TEST_CASE("implicit-transpose", "[matrix]")
{
    //implicit transposition is only allowed for vectors
    
    cuMat::VectorXiR v1(5);
    cuMat::VectorXiC v2a = v1;
    cuMat::VectorXiC v2b(v1);
    cuMat::VectorXiC v2c; v2c = v1;
    
    cuMat::RowVectorXiR v3(5);
    cuMat::RowVectorXiC v4a = v3;
    cuMat::RowVectorXiC v4b(v3);
    cuMat::RowVectorXiC v4c; v4c = v3;
    
    cuMat::ScalariR v5;
    cuMat::ScalariC v6a = v5;
    cuMat::ScalariC v6b(v5);
    cuMat::ScalariC v6c; v6c = v5;
    
    //This should not compile, how can I test that?
    //cuMat::MatrixXiR v7(5, 4);
    //cuMat::MatrixXiC v8a = v7;
    //cuMat::MatrixXiC v8b(v7);
    //cuMat::MatrixXiC v8c; v8c = v7;
}

//evalTo is deprecated
/*
// Matrix direct eval
template<typename T>
void testDirectEvalTo()
{
    typedef typename cuMat::Matrix<T, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> MatR;
    typedef typename cuMat::Matrix<T, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> MatC;
    T data[2][2][3] {
        {
            {1, 2, -1},
            {3, 4, -2}
        },
        {
            {5, 6, -3},
            {7, 8, -4}
        }
    };
    MatR matR = MatR::fromArray(data);
    MatC matC = matR+0;
    
    MatR targetR(2, 3, 2);
    MatC targetC(2, 3, 2);
    
    //memcpy
    targetR.setZero();
    targetC.setZero();
    CUMAT_PROFILING_RESET();
    matR.template evalTo<MatR, cuMat::AssignmentMode::ASSIGN>(targetR);
    matC.template evalTo<MatC, cuMat::AssignmentMode::ASSIGN>(targetC);
    REQUIRE(CUMAT_PROFILING_GET(EvalAny)==0);
    REQUIRE(CUMAT_PROFILING_GET(DeviceMemcpy)==2);
    assertMatrixEquality(matR, targetR);
    assertMatrixEquality(matC, targetC);
    
    //transpose
    targetR.setZero();
    targetC.setZero();
    CUMAT_PROFILING_RESET();
    matR.template evalTo<MatC, cuMat::AssignmentMode::ASSIGN>(targetC);
    matC.template evalTo<MatR, cuMat::AssignmentMode::ASSIGN>(targetR);
    REQUIRE(CUMAT_PROFILING_GET(EvalAny)==2);
    REQUIRE(CUMAT_PROFILING_GET(EvalTranspose) == (cuMat::internal::NumTraits<T>::IsCudaNumeric ? 2 : 0));
    REQUIRE(CUMAT_PROFILING_GET(EvalCwise) == (cuMat::internal::NumTraits<T>::IsCudaNumeric ? 0 : 2));
    REQUIRE(CUMAT_PROFILING_GET(DeviceMemcpy)==0);
    assertMatrixEquality(matR, targetR);
    assertMatrixEquality(matC, targetC);
    
    //cwise
    targetR.setZero();
    targetC.setZero();
    CUMAT_PROFILING_RESET();
    auto block1 = targetC.block(0,0,0,2,3,2);
    auto block2 = targetR.block(0,0,0,2,3,2);
    matR.template evalTo<decltype(block1), cuMat::AssignmentMode::ASSIGN>(block1);
    matC.template evalTo<decltype(block2), cuMat::AssignmentMode::ASSIGN>(block2);
    REQUIRE(CUMAT_PROFILING_GET(EvalAny)==2);
    REQUIRE(CUMAT_PROFILING_GET(EvalTranspose)==0);
    REQUIRE(CUMAT_PROFILING_GET(EvalCwise)==2);
    REQUIRE(CUMAT_PROFILING_GET(DeviceMemcpy)==0);
    assertMatrixEquality(matR, targetR);
    assertMatrixEquality(matC, targetC);
}
TEST_CASE("direct evalTo", "[matrix]")
{
    SECTION("int") {
        testDirectEvalTo<int>();
    }
    SECTION("float") {
        testDirectEvalTo<float>();
    }
    SECTION("double") {
        testDirectEvalTo<double>();    
    }
}
*/

//Deep clone
template<typename T>
void testDeepClone()
{
    typedef typename cuMat::Matrix<T, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::RowMajor> MatR;
    typedef typename cuMat::Matrix<T, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> MatC;
    T data[2][2][3] {
        {
            {1, 2, -1},
            {3, 4, -2}
        },
        {
            {5, 6, -3},
            {7, 8, -4}
        }
    };
    MatR matR = MatR::fromArray(data);
    MatC matC = matR+0;
    
    MatR cloneR1 = matR.deepClone();
    MatR cloneR2 = matR.template deepClone<cuMat::RowMajor>();
    MatC cloneR3 = matR.template deepClone<cuMat::ColumnMajor>();
    REQUIRE(matR.data() != cloneR1.data());
    REQUIRE(matR.data() != cloneR2.data());
    REQUIRE(matR.data() != cloneR3.data());
    assertMatrixEquality(matR, cloneR1);
    assertMatrixEquality(matR, cloneR2);
    assertMatrixEquality(matR, cloneR3);
    
    MatC cloneC1 = matC.deepClone();
    MatC cloneC2 = matC.template deepClone<cuMat::ColumnMajor>();
    MatR cloneC3 = matC.template deepClone<cuMat::RowMajor>();
    REQUIRE(matC.data() != cloneC1.data());
    REQUIRE(matC.data() != cloneC2.data());
    REQUIRE(matC.data() != cloneC3.data());
    assertMatrixEquality(matC, cloneC1);
    assertMatrixEquality(matC, cloneC2);
    assertMatrixEquality(matC, cloneC3);
}

TEST_CASE("deep clone", "[matrix]")
{
    SECTION("int") {
        testDeepClone<int>();
    }
    SECTION("float") {
        testDeepClone<float>();
    }
    SECTION("double") {
        testDeepClone<double>();    
    }
}
