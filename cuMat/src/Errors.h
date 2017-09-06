#ifndef __CUMAT_ERRORS_H__
#define __CUMAT_ERRORS_H__

#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <cstdarg>
#include <vector>

#include "Macros.h"
#include "Logging.h"

CUMAT_NAMESPACE_BEGIN

class cuda_error : public std::exception
{
private:
	std::string message_;
public:
	cuda_error(std::string message)
		: message_(message)
	{}

	const char* what() const throw() override
	{
		return message_.c_str();
	}
};

namespace internal {

	//enable verbose error checking (slow)
#ifndef CUMAT_VERBOSE_ERROR_CHECKING
#ifdef _DEBUG
#define CUMAT_VERBOSE_ERROR_CHECKING 1
#else
#define CUMAT_VERBOSE_ERROR_CHECKING 0
#endif
#endif

	class ErrorHelpers
	{
	public:
		static std::string vformat(const char *fmt, va_list ap)
		{
			// Allocate a buffer on the stack that's big enough for us almost
			// all the time.  Be prepared to allocate dynamically if it doesn't fit.
			size_t size = 1024;
			char stackbuf[1024];
			std::vector<char> dynamicbuf;
			char *buf = &stackbuf[0];
			va_list ap_copy;

			while (1) {
				// Try to vsnprintf into our buffer.
				va_copy(ap_copy, ap);
				int needed = vsnprintf(buf, size, fmt, ap);
				va_end(ap_copy);

				// NB. C99 (which modern Linux and OS X follow) says vsnprintf
				// failure returns the length it would have needed.  But older
				// glibc and current Windows return -1 for failure, i.e., not
				// telling us how much was needed.

				if (needed <= (int)size && needed >= 0) {
					// It fit fine so we're done.
					return std::string(buf, (size_t)needed);
				}

				// vsnprintf reported that it wanted to write more characters
				// than we allotted.  So try again using a dynamic buffer.  This
				// doesn't happen very often if we chose our initial size well.
				size = (needed > 0) ? (needed + 1) : (size * 2);
				dynamicbuf.resize(size);
				buf = &dynamicbuf[0];
			}
		}
		//Taken from https://stackoverflow.com/a/69911/4053176

		static std::string format(const char *fmt, ...)
		{
			va_list ap;
			va_start(ap, fmt);
			std::string buf = vformat(fmt, ap);
			va_end(ap);
			return buf;
		}
		//Taken from https://stackoverflow.com/a/69911/4053176

		// Taken from https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
		// and adopted

		static void cudaSafeCall(cudaError err, const char *file, const int line)
		{
	#ifndef NDEBUG
			if (cudaSuccess != err) {
				std::string msg = format("cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG(CUMAT_LOG_SEVERE) << msg;
				throw cuda_error(msg);
			}
	#if CUMAT_VERBOSE_ERROR_CHECKING==1
			//insert a device-sync
			err = cudaDeviceSynchronize();
			if (cudaSuccess != err) {
				std::string msg = format("cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG(CUMAT_LOG_SEVERE) << msg;
				throw cuda_error(msg);
			}
	#endif
	#endif
		}

		static void cudaCheckError(const char *file, const int line)
		{
#ifndef NDEBUG
			cudaError err = cudaGetLastError();
			if (cudaSuccess != err) {
				std::string msg = format("cudaCheckError() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG(CUMAT_LOG_SEVERE) << msg;
				throw cuda_error(msg);
			}

#if CUMAT_VERBOSE_ERROR_CHECKING==1
			// More careful checking. However, this will affect performance.
			err = cudaDeviceSynchronize();
			if (cudaSuccess != err) {
				std::string msg = format("cudaCheckError() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG(CUMAT_LOG_SEVERE) << msg;
				throw cuda_error(msg);
			}
#endif
#endif
		}
	};

#define CUMAT_SAFE_CALL( err ) CUMAT_NAMESPACE internal::ErrorHelpers::cudaSafeCall( err, __FILE__, __LINE__ )
#define CUMAT_CHECK_ERROR()    CUMAT_NAMESPACE internal::ErrorHelpers::cudaCheckError( __FILE__, __LINE__ )

}
CUMAT_NAMESPACE_END

#endif