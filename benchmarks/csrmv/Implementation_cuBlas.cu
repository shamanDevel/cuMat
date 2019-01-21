#include "benchmark.h"

#include <Eigen/Sparse>
#include <cuMat/Core>
#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include <cusparse_v2.h>

namespace {

//copied from cuMat/src/CublasApi.h

static const char* getErrorName(cusparseStatus_t status)
{
    switch (status)
    {
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED: cuSPARSE was not initialized";
    case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED: resource allocation failed";
    case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE: invalid value was passed as argument";
    case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH: device architecture not supported";
    case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR: access to GPU memory failed";
    case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED: general kernel launch failure";
    case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR: an internal error occured";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: functionality is not supported";
    case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT: pivot is zero";
    default: return "";
    }
}
static void cusparseSafeCall(cusparseStatus_t status, const char *file, const int line)
{
    if (CUBLAS_STATUS_SUCCESS != status) {
        std::string msg = cuMat::internal::ErrorHelpers::format("cusparseSafeCall() failed at %s:%i : %s\n",
            file, line, getErrorName(status));
        std::cerr << msg << std::endl;
        throw cuMat::cuda_error(msg);
    }
}
#define CUSPARSE_SAFE_CALL( err ) cusparseSafeCall( err, __FILE__, __LINE__ )

}

//Benchmark with cuBLAS
//cuMat is used to allocate the matrices, but the computation is done in cuBLAS (axpy)
void benchmark_cuBlas(
    const std::vector<std::string>& parameterNames,
    const Json::Array& parameters,
    const std::vector<std::string>& returnNames,
    Json::Array& returnValues)
{
	//number of runs for time measures
	const int runs = 10;
	const int subruns = 10;

	int numConfigs = parameters.Size();
	for (int config = 0; config < numConfigs; ++config)
	{
		//Input
		int gridSize = parameters[config][0].AsInt32();
		double totalTime = 0;
		std::cout << "  Grid Size: " << gridSize << std::flush;
		int matrixSize = gridSize * gridSize;

		//Create matrix
#define IDX(x, y) ((y) + (x)*gridSize)
		Eigen::SparseMatrix<float, Eigen::RowMajor, int> matrix(matrixSize, matrixSize);
		matrix.reserve(Eigen::VectorXi::Constant(matrixSize, 5));
		for (int x = 0; x<gridSize; ++x) for (int y = 0; y<gridSize; ++y)
		{
			int row = IDX(x, y);
			if (x > 0) matrix.insert(row, IDX(x - 1, y)) = -1;
			if (y > 0) matrix.insert(row, IDX(x, y - 1)) = -1;
			matrix.insert(row, row) = 4;
			if (y < gridSize - 1) matrix.insert(row, IDX(x, y + 1)) = -1;
			if (x < gridSize - 1) matrix.insert(row, IDX(x + 1, y)) = -1;
		}
		matrix.makeCompressed();

		//Create vector
		Eigen::VectorXf ex = Eigen::VectorXf::Random(matrixSize);
        
        //create cuBLAS handle
        cusparseHandle_t handle = nullptr;
        CUSPARSE_SAFE_CALL(cusparseCreate(&handle));

		//Copy to cuSparse
		int nnz = matrix.nonZeros();
		cuMat::VectorXi JA(matrixSize + 1); JA.copyFromHost(matrix.outerIndexPtr());
		cuMat::VectorXi IA(nnz); IA.copyFromHost(matrix.innerIndexPtr());
		cuMat::VectorXf data(nnz); data.copyFromHost(matrix.valuePtr());
		cuMat::VectorXf x = cuMat::VectorXf::fromEigen(ex);
		cuMat::VectorXf r(matrixSize);

		cusparseMatDescr_t matDescr = 0;
		const int* JAdata = JA.data();
		const int* IAdata = IA.data();
		const float* Adata = data.data();
		const float* xdata = x.data();
		float* rdata = r.data();
		CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&matDescr));
		cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

		float alpha = 1;
		float beta = 0;

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {
			CUMAT_SAFE_CALL(cudaMemsetAsync(rdata, 0, sizeof(float) * matrixSize));

            //Main logic
			cudaDeviceSynchronize();
			auto start = std::chrono::steady_clock::now();
            
            //pure cuBLAS + CUDA:

			for (int i = 0; i < subruns; ++i) {
				CUSPARSE_SAFE_CALL(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixSize, matrixSize, nnz, &alpha, matDescr,
					Adata, JAdata, IAdata, xdata, &beta, rdata));
			}

            cudaDeviceSynchronize();
			auto finish = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration_cast<
				std::chrono::duration<double>>(finish - start).count() * 1000 / subruns;

            totalTime += elapsed;
        }
        
        CUSPARSE_SAFE_CALL(cusparseDestroy(handle));

        //Result
        Json::Array result;
        double finalTime = totalTime / runs;
        result.PushBack(finalTime);
        returnValues.PushBack(result);
        std::cout << " -> " << finalTime << "ms" << std::endl;
    }
}
