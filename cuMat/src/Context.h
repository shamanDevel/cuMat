#ifndef __CUMAT_CONTEXT_H__
#define __CUMAT_CONTEXT_H__

#include <cuda_runtime.h>
#include <thread>
#include <mutex>

#include "Macros.h"
#include "Errors.h"
#include "Logging.h"

CUMAT_NAMESPACE_BEGIN

/**
 * \brief Stores the cuda context of the current thread.
 * cuMat uses one cuda stream per thread, and also potentially
 * different devices.
 * 
 */
struct Context
{
private:
	cudaStream_t stream_;
	int device_ = 0;

public:
	Context(int device = 0)
		: device_(device)
		, stream_(nullptr)
	{
#if 0
		//stream creation must be synchronized
		//TODO: seems to work without
		static std::mutex mutex;
		std::lock_guard<std::mutex> lock(mutex);
#endif

		CUMAT_SAFE_CALL(cudaStreamCreate(&stream_));

		//TODO: init BLAS context and so on

		CUMAT_LOG(CUMAT_LOG_DEBUG) << "Context initialized for thread 0x" << std::hex << std::this_thread::get_id()
		 << ", stream: 0x" << stream_;
	}

	~Context()
	{
		if (stream_ != nullptr)
		{
			CUMAT_SAFE_CALL(cudaStreamDestroy(stream_));
			stream_ = nullptr;
		}
		CUMAT_LOG(CUMAT_LOG_DEBUG) << "Context deleted for thread 0x" << std::hex << std::this_thread::get_id();
	}

	/**
	 * \brief Returns the cuda stream accociated with this context
	 * \return the cuda stream
	 */
	cudaStream_t stream() const { return stream_; }

	/**
	* \brief Returns the context of the current thread.
	* It is automatically created if not explicitly initialized with
	* assignDevice(int).
	* \return the current context.
	*/
	static Context& current()
	{
		static thread_local Context INSTANCE;
		return INSTANCE;
	}
};


CUMAT_NAMESPACE_END

#endif