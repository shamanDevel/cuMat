#ifndef __CUMAT_ERRORS_H__
#define __CUMAT_ERRORS_H__

#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <cstdarg>
#include <vector>
#include <stdexcept>
#include <stdio.h>

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
			if (cudaSuccess != err) {
				std::string msg = format("cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG_SEVERE(msg);
				throw cuda_error(msg);
			}
	#if CUMAT_VERBOSE_ERROR_CHECKING==1
			//insert a device-sync
			err = cudaDeviceSynchronize();
			if (cudaSuccess != err) {
				std::string msg = format("cudaSafeCall() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG_SEVERE(msg);
				throw cuda_error(msg);
			}
	#endif
		}

		static void cudaCheckError(const char *file, const int line)
		{
			cudaError err = cudaGetLastError();
			if (cudaSuccess != err) {
				std::string msg = format("cudaCheckError() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG_SEVERE(msg);
				throw cuda_error(msg);
			}

#if CUMAT_VERBOSE_ERROR_CHECKING==1
			// More careful checking. However, this will affect performance.
			err = cudaDeviceSynchronize();
			if (cudaSuccess != err) {
				std::string msg = format("cudaCheckError() failed at %s:%i : %s\n",
					file, line, cudaGetErrorString(err));
				CUMAT_LOG_SEVERE(msg);
				throw cuda_error(msg);
			}
#endif
		}
	};

/**
 * \brief Tests if the cuda library call wrapped inside the bracets was executed successfully, aka returned cudaSuccess
 * \param err the error code
 */
#define CUMAT_SAFE_CALL( err ) CUMAT_NAMESPACE internal::ErrorHelpers::cudaSafeCall( err, __FILE__, __LINE__ )
/**
 * \brief Issue this after kernel launches to check for errors in the kernel.
 */
#define CUMAT_CHECK_ERROR()    CUMAT_NAMESPACE internal::ErrorHelpers::cudaCheckError( __FILE__, __LINE__ )

	//TODO: find a better place in some Utility header

	/**
	 * \brief Numeric type conversion with overflow check.
	 * If <code>CUMAT_ENABLE_HOST_ASSERTIONS==1</code>, this method
	 * throws an std::runtime_error if the conversion results in
	 * an overflow.
	 * 
	 * If CUMAT_ENABLE_HOST_ASSERTIONS is not defined (default in release mode),
	 * this method simply becomes <code>static_cast</code>.
	 * 
	 * Source: The C++ Programming Language 4th Edition by Bjarne Stroustrup
	 * https://stackoverflow.com/a/30114062/1786598
	 * 
	 * \tparam Target the target type
	 * \tparam Source the source type
	 * \param v the source value
	 * \return the casted target value
	 */
	template<class Target, class Source>
	CUMAT_STRONG_INLINE Target narrow_cast(Source v)
	{
#if CUMAT_ENABLE_HOST_ASSERTIONS==1
		auto r = static_cast<Target>(v); // convert the value to the target type
		if (static_cast<Source>(r) != v)
			throw std::runtime_error("narrow_cast<>() failed");
		return r;
#else
		return static_cast<Target>(v);
#endif
	}

}
CUMAT_NAMESPACE_END

#endif