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
            if (CUBLAS_STATUS_SUCCESS != status) {
                std::string msg = ErrorHelpers::format("cublasSafeCall() failed at %s:%i : %s\n",
                    file, line, getErrorName(status));
				CUMAT_LOG_SEVERE(msg);
                throw cuda_error(msg);
            }
#if CUMAT_VERBOSE_ERROR_CHECKING==1
            //insert a device-sync
            cudaError err = cudaDeviceSynchronize();
            if (cudaSuccess != err) {
                std::string msg = ErrorHelpers::format("cublasSafeCall() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
				CUMAT_LOG_SEVERE(msg);
                throw cuda_error(msg);
            }
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
            CUBLAS_SAFE_CALL(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
            CUBLAS_SAFE_CALL(cublasSetStream(handle_, stream_));
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

        /**
         * \brief The complex types of cuMat (thrust::complex) and of cuBLAS (cuComplex=float2) are not
         * the same, but binary compatible. This function performs the cast
         * \tparam _Scalar the scalar type. No-op for non-complex types
         * \param p 
         * \return 
         */
        template<typename _Scalar> static _Scalar* cast(_Scalar* p) { return p; }
        static cuFloatComplex* cast(cfloat* p) { return reinterpret_cast<cuFloatComplex*>(p); }
        static const cuFloatComplex* cast(const cfloat* p) { return reinterpret_cast<const cuFloatComplex*>(p); }
        static cuDoubleComplex* cast(cdouble* p) { return reinterpret_cast<cuDoubleComplex*>(p); }
        static const cuDoubleComplex* cast(const cdouble* p) { return reinterpret_cast<const cuDoubleComplex*>(p); }

        //-------------------------------
        // MAIN API
        //-------------------------------

        //GEAM

#define CUBLAS_MAKE_WRAPPER(name, factory) \
    void cublas ## name ## Impl \
    factory(float, cublasS ## name) \
    void cublas ## name ## Impl \
    factory(double, cublasD ## name) \
    void cublas ## name ## Impl \
    factory(cuComplex, cublasC ## name) \
    void cublas ## name ## Impl \
    factory(cuDoubleComplex, cublasZ ## name)

#define CUBLAS_MAKE_WRAPPER_COMPLEX(name, factory) \
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


        //Normal GEMM

    private:
#define CUBLAS_GEMM_FACTORY(scalar, op) \
        (cublasOperation_t transA, cublasOperation_t transB, \
        int m, int n, int k,                                 \
        const scalar* alpha,                                 \
        const scalar* A, int lda,                            \
        const scalar* B, int ldb,                            \
        const scalar* beta,                                  \
        scalar* C, int ldc) {                                \
             CUBLAS_SAFE_CALL(op(handle_, transA, transB,    \
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)); \
        }
        CUBLAS_MAKE_WRAPPER(gemm, CUBLAS_GEMM_FACTORY)
#undef CUBLAS_GEMM_FACTORY
    public:
        /**
        * \brief Computes the matrix-matrix multiplication
        *        C = alpha op(A) * op(B) + beta C.
        * For details, see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
        * \tparam _Scalar the floating point scalar type
        */
        template <typename _Scalar>
        void cublasGemm(
            cublasOperation_t transA, cublasOperation_t transB,
            int m, int n, int k,
            const _Scalar* alpha,
            const _Scalar* A, int lda,
            const _Scalar* B, int ldb,
            const _Scalar* beta,
            _Scalar* C, int ldc)
        {
            cublasgemmImpl(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }

    private:
#define CUBLAS_GEMM3M_FACTORY(scalar, op) \
        (cublasOperation_t transA, cublasOperation_t transB, \
        int m, int n, int k,                                 \
        const scalar* alpha,                                 \
        const scalar* A, int lda,                            \
        const scalar* B, int ldb,                            \
        const scalar* beta,                                  \
        scalar* C, int ldc) {                                \
             CUBLAS_SAFE_CALL(op(handle_, transA, transB,    \
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)); \
        }
        CUBLAS_MAKE_WRAPPER_COMPLEX(gemm3m, CUBLAS_GEMM3M_FACTORY)
#undef CUBLAS_GEMM3M_FACTORY
    public:
        /**
        * \brief Computes the matrix-matrix multiplication using the Gauss-complexity reduction for complex matrices
        *        C = alpha op(A) * op(B) + beta C.
        * This is only supported on GPUs with architecture capabilities equal or greater than 5.0
        * For details, see http://docs.nvidia.com/cuda/cublas/index.html#cublas-gemm3m
        * \tparam _Scalar the complex floating point scalar type
        */
        template <typename _Scalar>
        void cublasGemm3m(
            cublasOperation_t transA, cublasOperation_t transB,
            int m, int n, int k,
            const _Scalar* alpha,
            const _Scalar* A, int lda,
            const _Scalar* B, int ldb,
            const _Scalar* beta,
            _Scalar* C, int ldc)
        {
            cublasgemm3mImpl(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }


        //Batched GEMM

    private:
#define CUBLAS_BATCHED_GEMM_FACTORY(scalar, op) \
        (cublasOperation_t transA, cublasOperation_t transB, \
        int m, int n, int k,                                 \
        const scalar* alpha,                                 \
        const scalar* A, int lda,                            \
        long long int strideA,                               \
        const scalar* B, int ldb,                            \
        long long int strideB,                               \
        const scalar* beta,                                  \
        scalar* C, int ldc,                                  \
        long long int strideC,                               \
        int batchCount)                                      \
        {                                                    \
             CUBLAS_SAFE_CALL(op(handle_, transA, transB,    \
                m, n, k,                                     \
                alpha, A, lda, strideA, B, ldb, strideB,     \
                beta, C, ldc, strideC, batchCount));         \
        }
        CUBLAS_MAKE_WRAPPER(gemmStridedBatched, CUBLAS_BATCHED_GEMM_FACTORY)
#undef CUBLAS_BATCHED_GEMM_FACTORY
    public:
        /**
        * \brief Computes the batched matrix-matrix multiplication
        *        C = alpha op(A) * op(B) + beta C.
        * For details, see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmstridedbatched
        * \tparam _Scalar the floating point scalar type
        */
        template <typename _Scalar>
        void cublasGemmBatched(
            cublasOperation_t transA, cublasOperation_t transB,
            int m, int n, int k,
            const _Scalar* alpha,
            const _Scalar* A, int lda,
            long long int strideA,
            const _Scalar* B, int ldb,
            long long int strideB,
            const _Scalar* beta,
            _Scalar* C, int ldc,
            long long int strideC,
            int batchCount)
        {
            cublasgemmStridedBatchedImpl(transA, transB, m, n, k, 
                alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        }


#undef CUBLAS_MAKE_WRAPPER
#undef CUBLAS_MAKE_WRAPPER_COMPLEX
#undef CUBLAS_SAFE_CALL
    };

}

CUMAT_NAMESPACE_END


#endif
