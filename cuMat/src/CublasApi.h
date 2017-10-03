#ifndef __CUMAT_CUBLAS_API_H__
#define __CUMAT_CUBLAS_API_H__

#include <cublas_v2.h>
#include <string>

#include "Macros.h"
#include "Errors.h"
#include "Logging.h"
#include "Context.h"
#include "NumTraits.h"
#include "DevicePointer.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{

    /**
     * \brief Interface class to cuBLAS.
     * Note that cuBLAS assumes all matrices to be in ColumnMajor order.
     */
    class CublasApi
    {
    private:
        cublasHandle_t handle_;
        cudaStream_t stream_;

    private:

        //-------------------------------
        // ERROR HANDLING
        //-------------------------------

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
#ifndef NDEBUG
            if (CUBLAS_STATUS_SUCCESS != status) {
                std::string msg = ErrorHelpers::format("cublasSafeCall() failed at %s:%i : %s\n",
                    file, line, getErrorName(status));
                CUMAT_LOG(CUMAT_LOG_SEVERE) << msg;
                throw cuda_error(msg);
            }
#if CUMAT_VERBOSE_ERROR_CHECKING==1
            //insert a device-sync
            cudaError err = cudaDeviceSynchronize();
            if (cudaSuccess != err) {
                std::string msg = ErrorHelpers::format("cublasSafeCall() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
                CUMAT_LOG(CUMAT_LOG_SEVERE) << msg;
                throw cuda_error(msg);
            }
#endif
#endif
        }
#define CUBLAS_SAFE_CALL( err ) cublasSafeCall( err, __FILE__, __LINE__ )

        //-------------------------------
        // CREATION
        //-------------------------------

        CUMAT_DISALLOW_COPY_AND_ASSIGN(CublasApi);
    private:
        CublasApi(Context& ctx)
            : handle_(nullptr)
            , stream_(ctx.stream())
        {
            CUBLAS_SAFE_CALL(cublasCreate_v2(&handle_));
        }
    public:
        ~CublasApi()
        {
            if (handle_ != nullptr)
                CUBLAS_SAFE_CALL(cublasDestroy_v2(handle_));
        }
        /**
         * \brief Returns the cuBLAS wrapper bound to the current instance.
         * \return the cuBLAS wrapper
         */
        static CublasApi& current()
        {
            static thread_local CublasApi INSTANCE(Context::current());
            return INSTANCE;
        }

        //-------------------------------
        // MAIN API
        //-------------------------------

#define CUBLAS_MAKE_WRAPPER(name, factory) \
    void cublas ## name ## Impl \
    factory(float, cublasS ## name) \
    void cublas ## name ## Impl \
    factory(double, cublasD ## name) \
    void cublas ## name ## Impl \
    factory(cuComplex, cublasC ## name) \
    void cublas ## name ## Impl \
    factory(cuDoubleComplex, cublasZ ## name)

    private:
#define CUBLAS_GEAM_FACTORY(scalar, op) \
        (cublasOperation_t transA, cublasOperation_t transB, \
        int m, int n,                                        \
        const scalar* alpha,                                 \
        const scalar* A, int lda,                            \
        const scalar* beta,                                  \
        const scalar* B, int ldb,                            \
        scalar* C, int ldc) {                                \
             CUBLAS_SAFE_CALL(op(handle_, transA, transB,    \
                m, n, alpha, A, lda, beta, B, ldb, C, ldc)); \
        }
        CUBLAS_MAKE_WRAPPER(geam, CUBLAS_GEAM_FACTORY)
#undef CUBLAS_GEAM_FACTORY
    public:
        /**
         * \brief Computes the matrix-matrix addition/transposition
         *        C = alpha op(A) + beta op(B).
         * For details, see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam
         * \tparam _Scalar the floating point scalar type
         */
        template <typename _Scalar>
        void cublasGeam(
            cublasOperation_t transA, cublasOperation_t transB,
            int m, int n,
            const _Scalar* alpha,
            const _Scalar* A, int lda,
            const _Scalar* beta,
            const _Scalar* B, int ldb,
            _Scalar* C, int ldc)
        {
            cublasgeamImpl(transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
        }

#undef CUBLAS_MAKE_WRAPPER
#undef CUBLAS_SAFE_CALL
    };

}

CUMAT_NAMESPACE_END


#endif