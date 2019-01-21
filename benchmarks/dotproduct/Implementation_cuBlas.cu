#include "benchmark.h"

#include <cuMat/Core>
#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>

namespace {

//copied from cuMat/src/CublasApi.h

static const char* getErrorName(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED: cuBLAS was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED: resource allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE: invalid value was passed as argument";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH: device architecture not supported";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR: access to GPU memory failed";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED: general kernel launch failure";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR: an internal error occured";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED: functionality is not supported";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR: required licence was not found";
    default: return "";
    }
}
static void cublasSafeCall(cublasStatus_t status, const char *file, const int line)
{
    if (CUBLAS_STATUS_SUCCESS != status) {
        std::string msg = cuMat::internal::ErrorHelpers::format("cublasSafeCall() failed at %s:%i : %s\n",
            file, line, getErrorName(status));
        std::cerr << msg << std::endl;
        throw cuMat::cuda_error(msg);
    }
}
#define CUBLAS_SAFE_CALL( err ) cublasSafeCall( err, __FILE__, __LINE__ )

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

    //test if the config is valid
    assert(parameterNames.size() == 1);
    assert(parameterNames[0] == "Vector-Size");
    assert(returnNames.size() == 1);
    assert(returnNames[0] == "Time");

    cuMat::SimpleRandom rand;

    int numConfigs = parameters.Size();
    for (int config = 0; config < numConfigs; ++config)
    {
		//Input
		int vectorSize = parameters[config][0].AsInt32();
		double totalTime = 0;
		std::cout << "  VectorSize: " << vectorSize << std::flush;

		//Create matrices
		cuMat::VectorXf a(vectorSize); rand.fillUniform(a, 0, 1);
		cuMat::VectorXf b(vectorSize); rand.fillUniform(b, 0, 1);
		float* aRaw = a.data();
		float* bRaw = b.data();
        
        //create cuBLAS handle
        cublasHandle_t handle = nullptr;
        CUBLAS_SAFE_CALL(cublasCreate_v2(&handle));
		CUBLAS_SAFE_CALL(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));

        //Run it multiple times
        for (int run = 0; run < runs; ++run)
        {

            //Main logic
			cudaDeviceSynchronize();
			auto start = std::chrono::steady_clock::now();
            
            //pure cuBLAS + CUDA:
			for (int subrun = 0; subrun < subruns; ++subrun) {
				float result;
				CUBLAS_SAFE_CALL(cublasSdot_v2(handle, vectorSize, aRaw, 1, bRaw, 1, &result));
			}

            cudaDeviceSynchronize();
			auto finish = std::chrono::steady_clock::now();
			double elapsed = std::chrono::duration_cast<
				std::chrono::duration<double>>(finish - start).count() * 1000 / subruns;

            totalTime += elapsed;
        }
        
        CUBLAS_SAFE_CALL(cublasDestroy_v2(handle));

        //Result
        Json::Array result;
        double finalTime = totalTime / runs;
        result.PushBack(finalTime);
        returnValues.PushBack(result);
        std::cout << " -> " << finalTime << "ms" << std::endl;
    }
}
