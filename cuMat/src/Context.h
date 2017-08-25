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
	 * \brief Allocates size-number of bytes on the host system.
	 * This memory must be freed with freeHost(void*).
	 * 
	 * Note that allocate zero bytes is perfectly fine. 
	 * That is also the only case in which this function might
	 * return NULL.
	 * Otherwise, always a valid pointer must be returned
	 * or an exception is thrown.
	 * 
	 * \param size the number of bytes to allocate
	 * \return the adress of the new host memory
	 */
	void* mallocHost(size_t size)
	{
		//TODO: add a plugin-mechanism for custom allocators
		if (size == 0) return nullptr;
		void* memory;
		CUMAT_SAFE_CALL(cudaMallocHost(&memory, size));
		return memory;
	}

	/**
	* \brief Allocates size-number of bytes on the device system.
	* This memory must be freed with freeDevice(void*).
	*
	* Note that allocate zero bytes is perfectly fine.
	* That is also the only case in which this function might
	* return NULL.
	* Otherwise, always a valid pointer must be returned
	* or an exception is thrown.
	*
	* \param size the number of bytes to allocate
	* \return the adress of the new device memory
	*/
	void* mallocDevice(size_t size)
	{
		//TODO: add a plugin-mechanism for custom allocators
		if (size == 0) return nullptr;
		void* memory;
		CUMAT_SAFE_CALL(cudaMalloc(&memory, size));
		return memory;
	}

	/**
	 * \brief Frees memory previously allocated with allocateHost(size_t).
	 * Passing a NULL-pointer should be a no-op.
	 * \param memory the memory to be freed
	 */
	void freeHost(void* memory)
	{
		//TODO: add a plugin-mechanism for custom allocators
		CUMAT_SAFE_CALL(cudaFreeHost(memory));
	}

	/**
	* \brief Frees memory previously allocated with allocateDevice(size_t).
	* Passing a NULL-pointer should be a no-op.
	* \param memory the memory to be freed
	*/
	void freeDevice(void* memory)
	{
		//TODO: add a plugin-mechanism for custom allocators
		CUMAT_SAFE_CALL(cudaFree(memory));
	}

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