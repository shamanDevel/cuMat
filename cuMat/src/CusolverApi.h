#ifndef __CUMAT_CUSOLVER_API_H__
#define __CUMAT_CUSOLVER_API_H__

#include <cusolverDn.h>
#include <string>

#include "Macros.h"
#include "Errors.h"
#include "Logging.h"
#include "Context.h"
#include "NumTraits.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{

    /**
     * \brief Interface class to cuSOLVER.
     * Note that cuSOLVER assumes all matrices to be in ColumnMajor order.
     */
    class CusolverApi
    {
    private:
        Context* ctx_;
        cusolverDnHandle_t handle_;
        cudaStream_t stream_;

    private:

        //-------------------------------
        // ERROR HANDLING
        //-------------------------------

        static const char* getErrorName(cusolverStatus_t status)
        {
            switch (status)
            {
            case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED: cuSOLVER was not initialized";
            case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED: resource allocation failed";
            case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE: invalid value was passed as argument";
            case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH: device architecture not supported";
            case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED: general kernel launch failure";
            case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR: an internal error occured";
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: matrix type is not supported";
            default: return "";
            }
        }
        static void cusolverSafeCall(cusolverStatus_t status, const char *file, const int line)
        {
            if (CUSOLVER_STATUS_SUCCESS != status) {
                std::string msg = ErrorHelpers::format("cusolverSafeCall() failed at %s:%i : %s\n",
                    file, line, getErrorName(status));
				CUMAT_LOG_SEVERE(msg);
                throw cuda_error(msg);
            }
#if CUMAT_VERBOSE_ERROR_CHECKING==1
            //insert a device-sync
            cudaError err = cudaDeviceSynchronize();
            if (cudaSuccess != err) {
                std::string msg = ErrorHelpers::format("cusolverSafeCall() failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString(err));
				CUMAT_LOG_SEVERE(msg);
                throw cuda_error(msg);
            }
#endif
        }
#define CUSOLVER_SAFE_CALL( err ) cusolverSafeCall( err, __FILE__, __LINE__ )

        //-------------------------------
        // CREATION
        //-------------------------------

        CUMAT_DISALLOW_COPY_AND_ASSIGN(CusolverApi);
    private:
        CusolverApi(Context& ctx)
            : ctx_(&ctx)
            , handle_(nullptr)
            , stream_(ctx.stream())
        {
            CUSOLVER_SAFE_CALL(cusolverDnCreate(&handle_));
            CUSOLVER_SAFE_CALL(cusolverDnSetStream(handle_, stream_));
        }
    public:
        ~CusolverApi()
        {
            if (handle_ != nullptr)
                CUSOLVER_SAFE_CALL(cusolverDnDestroy(handle_));
        }
        /**
         * \brief Returns the cuBLAS wrapper bound to the current instance.
         * \return the cuBLAS wrapper
         */
        static CusolverApi& current()
        {
            static thread_local CusolverApi INSTANCE(Context::current());
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

#define CUSOLVER_MAKE_WRAPPER(name, factory) \
    void name ## Impl \
    factory(float, cusolverDnS ## name) \
    void name ## Impl \
    factory(double, cusolverDnD ## name) \
    void name ## Impl \
    factory(cuComplex, cusolverDnC ## name) \
    void name ## Impl \
    factory(cuDoubleComplex, cusolverDnZ ## name)

        //POTRF

    private:
#define CUSOLVER_POTRF_FACTORY(scalar, op)                                                          \
        (cublasFillMode_t uplo,                                                                     \
        int n,                                                                                      \
        scalar* A, int lda,                                                                         \
        int* devInfo                                                                                \
        ) {                                                                                         \
            int Lwork;                                                                              \
            CUSOLVER_SAFE_CALL(op ## _bufferSize(handle_, uplo, n, A, lda, &Lwork));                \
            scalar* workspace = static_cast<scalar*>(ctx_->mallocDevice(sizeof(scalar) * Lwork));   \
            CUSOLVER_SAFE_CALL(op(handle_, uplo, n, A, lda, workspace, Lwork, devInfo));            \
            ctx_->freeDevice(workspace);                                                            \
        }
        CUSOLVER_MAKE_WRAPPER(potrf, CUSOLVER_POTRF_FACTORY)
#undef CUSOLVER_POTRF_FACTORY
    public:
        /**
         * \brief This function computes the Cholesky factorization of a Hermitian positive-definite matrix.
         * For details, see http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrf
         * \tparam _Scalar the floating point scalar type
         */
        template <typename _Scalar>
        void cusolverPotrf(
            cublasFillMode_t uplo,
            int n,
            _Scalar* A, int lda,
            int *devInfo)
        {
            potrfImpl(uplo, n, A, lda, devInfo);
        }

        // POTRS

    private:
#define CUSOLVER_POTRS_FACTORY(scalar, op)                                              \
        (cublasFillMode_t uplo,                                                         \
        int n, int nrhs,                                                                \
        const scalar* A, int lda,                                                       \
        scalar* B, int ldb,                                                             \
        int* devInfo                                                                    \
        ) {                                                                             \
            CUSOLVER_SAFE_CALL(op(handle_, uplo, n, nrhs, A, lda, B, ldb, devInfo));    \
        }
        CUSOLVER_MAKE_WRAPPER(potrs, CUSOLVER_POTRS_FACTORY)
#undef CUSOLVER_POTRS_FACTORY
    public:
        /**
        * \brief This function solves a system of linear equations AX=B where A is a n×n Hermitian matrix, only lower or upper part is meaningful.
        * For details, see http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs
        * \tparam _Scalar the floating point scalar type
        */
        template <typename _Scalar>
        void cusolverPotrs(
            cublasFillMode_t uplo,
            int n,
            int nrhs,
            const _Scalar *A,
            int lda,
            _Scalar *B,
            int ldb,
            int *devInfo)
        {
            potrsImpl(uplo, n, nrhs, A, lda, B, ldb, devInfo);
        }


        //GETRF

    private:
#define CUSOLVER_GETRF_FACTORY(scalar, op)                                                          \
        (int m, int n,                                                                              \
        scalar* A, int lda,                                                                         \
        int* devIpiv, int* devInfo                                                                  \
        ) {                                                                                         \
            int Lwork;                                                                              \
            CUSOLVER_SAFE_CALL(op ## _bufferSize(handle_, m, n, A, lda, &Lwork));                   \
            scalar* workspace = static_cast<scalar*>(ctx_->mallocDevice(sizeof(scalar) * Lwork));   \
            CUSOLVER_SAFE_CALL(op(handle_, m, n, A, lda, workspace, devIpiv, devInfo));             \
            ctx_->freeDevice(workspace);                                                            \
        }
        CUSOLVER_MAKE_WRAPPER(getrf, CUSOLVER_GETRF_FACTORY)
#undef CUSOLVER_GETRF_FACTORY
    public:
        /**
        * \brief This function computes the LU factorization of a m×n matrix P*A=L*U.
        * For details, see http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf
        * \tparam _Scalar the floating point scalar type
        */
        template <typename _Scalar>
        void cusolverGetrf(
            int m, int n,
            _Scalar* A, int lda,
            int* devIpiv,
            int *devInfo)
        {
            getrfImpl(m, n, A, lda, devIpiv, devInfo);
        }

        // GETRS

    private:
#define CUSOLVER_GETRS_FACTORY(scalar, op)                                              \
        (cublasOperation_t trans,                                                       \
        int n, int nrhs,                                                                \
        const scalar* A, int lda,                                                       \
        const int *devIpiv,                                                             \
        scalar* B, int ldb,                                                             \
        int* devInfo                                                                    \
        ) {                                                                             \
            CUSOLVER_SAFE_CALL(op(handle_, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));    \
        }
        CUSOLVER_MAKE_WRAPPER(getrs, CUSOLVER_GETRS_FACTORY)
#undef CUSOLVER_GETRS_FACTORY
    public:
        /**
        * \brief This function solves a system of linear equations op(A)*X=B where A is a nxn matrix that was LU-factored by getrf.
        * For details, see http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrs
        * \tparam _Scalar the floating point scalar type
        */
        template <typename _Scalar>
        void cusolverGetrs(
            cublasOperation_t trans,
            int n,
            int nrhs,
            const _Scalar *A,
            int lda,
            const int *devIpiv,
            _Scalar *B,
            int ldb,
            int *devInfo)
        {
            getrsImpl(trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
        }

#undef CUSOLVER_MAKE_WRAPPER
#undef CUSOLVER_SAFE_CALL
    };

}

CUMAT_NAMESPACE_END


#endif
