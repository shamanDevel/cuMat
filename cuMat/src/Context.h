#ifndef __CUMAT_CONTEXT_H__
#define __CUMAT_CONTEXT_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thread>
#include <mutex>
#include <algorithm>
#include <typeinfo>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Errors.h"
#include "Logging.h"
#include "Profiling.h"

#ifndef CUMAT_SINGLE_THREAD_CONTEXT
/**
 * \brief Specifies whether Context::current() returns 
 *  - a separate context for each thread (0), the default
 *  - a global context, shared over all threads (1)
 */
#define CUMAT_SINGLE_THREAD_CONTEXT 0
#endif

#ifndef CUMAT_CONTEXT_DEBUG_MEMORY
/**
 * \brief Define this constant as 1 to enable a simple mechanism to test for memory leaks
 */
#define CUMAT_CONTEXT_DEBUG_MEMORY 0
#else
#if defined(NDEBUG) && CUMAT_CONTEXT_DEBUG_MEMORY==1
#error You requested to turn on CUMAT_CONTEXT_DEBUG_MEMORY but disabled assertions
#endif
#endif
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
#include <assert.h>
#endif


/**
 * \brief If this macro is defined to 1, then the cub cached allocator is used for device allocation.
 */
#define CUMAT_CONTEXT_USE_CUB_ALLOCATOR 1

#if CUMAT_CONTEXT_USE_CUB_ALLOCATOR==1
//#include <cub/util_allocator.cuh>
#include "../../third-party/cub/util_allocator.cuh" //No need to add cub to the global include (would clash e.g. with other Eigen versions)
#endif

CUMAT_NAMESPACE_BEGIN

/**
 * \brief A structure holding information for launching
 * a 1D, 2D or 3D kernel.
 * 
 * Sample code for kernels:
 * \code
 *     __global__ void My1DKernel(dim3 virtual_size, ...) {
 *         CUMAT_KERNEL_1D_LOOP(i, virtual_size) {
 *             // do something at e.g. matrix.rawCoeff(i)
 *         }
 *     }
 *     
 *     __global__ void My2DKernel(dim3 virtual_size, ...) {
 *         CUMAT_KERNEL_2D_LOOP(i, j, virtual_size) {
 *             // do something at e.g. matrix.coeff(i, j, 0)
 *         }
 *     }
 *     
 *     __global__ void My3DKernel(dim3 virtual_size, ...) {
 *         CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size) {
 *             // do something at e.g. matrix.coeff(i, j, k)
 *         }
 *     }
 * \endcode
 * 
 * Launch the 1D,2D,3D-kernel using:
 * \code
 *     KernelLaunchConfig cfg = context.createLaunchConfigXD(...);
 *     MyKernel<<<cfg.block_count, cfg.thread_per_block, 0, context.stream()>>>(cfg.virtual_size, ...);
 * \endcode
 */
struct KernelLaunchConfig
{
	dim3 virtual_size;
	dim3 thread_per_block;
	dim3 block_count;
};

/**
 * \brief 1D-loop over the jobs in a kernel.
 * Has to be closed with CUMAT_KERNEL_1D_LOOP_END
 */
#define CUMAT_KERNEL_1D_LOOP(i, virtual_size)								\
	for (CUMAT_NAMESPACE Index i = blockIdx.x * blockDim.x + threadIdx.x;	\
		 i < virtual_size.x;												\
		 i += blockDim.x * gridDim.x) {
#define CUMAT_KERNEL_1D_LOOP_END }

/**
 * \brief 2D-loop over the jobs in a kernel, coordinate i runs fastest.
 * Has to be closed with CUMAT_KERNEL_2D_LOOP_END
 */
#define CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)							\
	for (CUMAT_NAMESPACE Index __i = blockIdx.x * blockDim.x + threadIdx.x;	\
		 __i < virtual_size.x*virtual_size.y;								\
		 __i += blockDim.x * gridDim.x) {									\
		 CUMAT_NAMESPACE Index j = __i / virtual_size.x;					\
		 CUMAT_NAMESPACE Index i = __i - j * virtual_size.x;
#define CUMAT_KERNEL_2D_LOOP_END }

 /**
 * \brief 3D-loop over the jobs in a kernel, coordinate i runs fastest, followed by j.
 * Has to be closed with CUMAT_KERNEL_3D_LOOP_END
 */
#define CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size) 												\
	for (CUMAT_NAMESPACE Index __i = blockIdx.x * blockDim.x + threadIdx.x;							\
		 __i < virtual_size.x*virtual_size.y*virtual_size.z;										\
		 __i += blockDim.x * gridDim.x) {															\
		 CUMAT_NAMESPACE Index k = __i / (virtual_size.x*virtual_size.y);							\
		 CUMAT_NAMESPACE Index j = (__i - (k * virtual_size.x*virtual_size.y)) / virtual_size.x;	\
		 CUMAT_NAMESPACE Index i = __i - virtual_size.x * (j + virtual_size.y * k);
#define CUMAT_KERNEL_3D_LOOP_END }

/**
 * \brief Stores the cuda context of the current thread.
 * cuMat uses one cuda stream per thread, and also potentially
 * different devices.
 * 
 */
class Context
{
private:
	cudaStream_t stream_;
	int device_ = 0;

#if CUMAT_CONTEXT_DEBUG_MEMORY==1
	int allocationsHost_ = 0;
	int allocationsDevice_ = 0;
#endif

	CUMAT_DISALLOW_COPY_AND_ASSIGN(Context);

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

		CUMAT_SAFE_CALL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

		//TODO: init BLAS context and so on

		CUMAT_LOG_DEBUG("Context initialized for thread 0x" << std::hex << std::this_thread::get_id() << ", stream: 0x" << stream_);
	}

	~Context()
	{
		if (stream_ != nullptr)
		{
			CUMAT_SAFE_CALL(cudaStreamDestroy(stream_));
			stream_ = nullptr;
		}
		CUMAT_LOG_DEBUG("Context deleted for thread 0x" << std::hex << std::this_thread::get_id());
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
		CUMAT_ASSERT(allocationsHost_ == 0 && "some host memory was not released");
		CUMAT_ASSERT(allocationsDevice_ == 0 && "some device memory was not released");
#endif
	}

	/**
	* \brief Returns the context of the current thread.
	* It is automatically created if not explicitly initialized with
	* assignDevice(int).
	* \return the current context.
	*/
	static Context& current()
	{
#if CUMAT_SINGLE_THREAD_CONTEXT==0
		//per thread instance
		static thread_local Context INSTANCE;
#else
		//global instance
		static Context INSTANCE;
#endif
		return INSTANCE;
	}

	/**
	 * \brief Returns the cuda stream accociated with this context
	 * \return the cuda stream
	 */
	cudaStream_t stream() const { return stream_; }

#if CUMAT_CONTEXT_USE_CUB_ALLOCATOR==1
    static cub::CachingDeviceAllocator& getCubAllocator()
	{
	    //the allocator is shared over all devices and threads for best caching behavior
        //Cub synchronizes the access internally
        static cub::CachingDeviceAllocator INSTANCE;
        return INSTANCE;
	}
#endif

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
        CUMAT_PROFILING_INC(HostMemAlloc);
		//TODO: add a plugin-mechanism for custom allocators
		if (size == 0) return nullptr;
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
		allocationsHost_++;
#endif
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
        CUMAT_PROFILING_INC(DeviceMemAlloc);
		//TODO: add a plugin-mechanism for custom allocators
		if (size == 0) return nullptr;
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
		allocationsDevice_++;
#endif
		void* memory;
#if CUMAT_CONTEXT_USE_CUB_ALLOCATOR==1
        CUMAT_SAFE_CALL(getCubAllocator().DeviceAllocate(device_, &memory, size, stream_));
#else
		CUMAT_SAFE_CALL(cudaMalloc(&memory, size));
#endif
		return memory;
	}

	/**
	 * \brief Frees memory previously allocated with allocateHost(size_t).
	 * Passing a NULL-pointer should be a no-op.
	 * \param memory the memory to be freed
	 */
	void freeHost(void* memory)
	{
        CUMAT_PROFILING_INC(HostMemFree);
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
		if (memory != nullptr) {
			allocationsHost_--;
			CUMAT_ASSERT(allocationsHost_ >= 0 && "You freed more pointers than were allocated");
		}
#endif
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
        CUMAT_PROFILING_INC(DeviceMemFree);
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
		if (memory != nullptr) {
			allocationsDevice_--;
			CUMAT_ASSERT(allocationsDevice_ >= 0 && "You freed more pointers than were allocated");
	}
#endif
#if CUMAT_CONTEXT_USE_CUB_ALLOCATOR==1
        if (memory != nullptr) {
            CUMAT_SAFE_CALL(getCubAllocator().DeviceFree(device_, memory));
        }
#else
		CUMAT_SAFE_CALL(cudaFree(memory));
#endif
	}
	
#if CUMAT_CONTEXT_DEBUG_MEMORY==1
	//For testing only
	int getAliveHostPointers() const { return allocationsHost_; }
	//For testing only
	int getAliveDevicePointers() const { return allocationsDevice_; }
#endif

    /**
     * \brief Returns the free memory on the device in bytes.
     * \return the free memory in bytes
     */
    static size_t getFreeDeviceMemory()
	{
        size_t free, total;
        CUMAT_SAFE_CALL(cudaMemGetInfo(&free, &total));
        return free;
	}

    /**
    * \brief Returns the total available memory on the device in bytes.
    * \return the total available memory in bytes
    */
    static size_t getTotalDeviceMemory()
    {
        size_t free, total;
        CUMAT_SAFE_CALL(cudaMemGetInfo(&free, &total));
        return total;
    }

	/**
	 * \brief Returns the kernel launch configurations for a 1D launch.
	 * For details on how to use it, see the documentation of
	 * KernelLaunchConfig.
	 * \param size the size of the problem
	 * \param func the device function
	 */
	template<class T>
	KernelLaunchConfig createLaunchConfig1D(Index size, T func) const
	{
		const unsigned int size_ = static_cast<unsigned int>(size);
		CUMAT_ASSERT_ARGUMENT(size > 0);
		CUMAT_ASSERT(Index(size_) == size && "size exceeds the range of unsigned int!");
#if 0
		//Very simplistic first version
		unsigned int blockSize = 256u;
		KernelLaunchConfig cfg = {
			dim3(size, 1, 1),
			dim3(blockSize, 1, 1),
			dim3(CUMAT_DIV_UP(size, blockSize), 1, 1)
		};
		return cfg;
#else
		//Improved version using cudaOccupancyMaxPotentialBlockSize
		int minGridSize = 0, bestBlockSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, func);
		minGridSize = std::min(int(CUMAT_DIV_UP(size_, bestBlockSize)), minGridSize);
		CUMAT_LOG_DEBUG("Best potential occupancy for " << typeid(T).name() << " found to be: blocksize=" << bestBlockSize << ", gridSize=" << minGridSize);
		KernelLaunchConfig cfg = {
			dim3(size_, 1, 1),
			dim3(bestBlockSize, 1, 1),
			dim3(minGridSize, 1, 1)
		};
		return cfg;
#endif
	}

	/**
	* \brief Returns the kernel launch configurations for a 2D launch.
	* For details on how to use it, see the documentation of
	* KernelLaunchConfig.
	* \param sizex the size of the problem along x
	* \param sizey the size of the problem along y
	* \param func the kernel function
	* \return the launch configuration
	*/
	template<class T>
	KernelLaunchConfig createLaunchConfig2D(unsigned int sizex, unsigned int sizey, T func) const
	{
		CUMAT_ASSERT_ARGUMENT(sizex > 0);
		CUMAT_ASSERT_ARGUMENT(sizey > 0);
		int minGridSize = 0, bestBlockSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, func);
		CUMAT_LOG_DEBUG("Best potential occupancy for " << typeid(T).name() << " found to be: blocksize=" << bestBlockSize << ", gridSize=" << minGridSize);
		KernelLaunchConfig cfg = {
			dim3(sizex, sizey, 1),
			dim3(bestBlockSize, 1, 1),
			dim3(minGridSize, 1, 1)
		};
		return cfg;
	}

	/**
	* \brief Returns the kernel launch configurations for a 3D launch.
	* For details on how to use it, see the documentation of
	* KernelLaunchConfig.
	* \param sizex the size of the problem along x
	* \param sizey the size of the problem along y
	* \param sizez the size of the problem along z
	* \param func the kernel function
	* \return the launch configuration
	*/
	template<class T>
	KernelLaunchConfig createLaunchConfig3D(unsigned int sizex, unsigned int sizey, unsigned int sizez, T func) const
	{
		CUMAT_ASSERT_ARGUMENT(sizex > 0);
		CUMAT_ASSERT_ARGUMENT(sizey > 0);
		CUMAT_ASSERT_ARGUMENT(sizez > 0);
		int minGridSize = 0, bestBlockSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, func);
		CUMAT_LOG_DEBUG("Best potential occupancy for " << typeid(T).name() << " found to be: blocksize=" << bestBlockSize << ", gridSize=" << minGridSize);
		KernelLaunchConfig cfg = {
			dim3(sizex, sizey, sizez),
			dim3(bestBlockSize, 1, 1),
			dim3(minGridSize, 1, 1)
		};
		return cfg;
	}
};

/**
 * \brief A simple reference-counted wrapper around cuda events.
 * This class is not synchronized
 */
class Event
{
	cudaEvent_t event_;
	size_t* counter_;

		void release()
	{
		if ((counter_) && (--(*counter_) == 0))
		{
			delete counter_;
			CUMAT_SAFE_CALL(cudaEventDestroy(event_));
		}
	}

public:
	Event()
		: event_(nullptr)
		, counter_(new size_t(1))
	{
		CUMAT_SAFE_CALL(cudaEventCreate(&event_));
	}

	Event(const Event& rhs)
		: event_(rhs.event_)
		, counter_(rhs.counter_)
	{
		assert(counter_);
		++(*counter_);
	}

	Event(Event&& rhs) noexcept
		: event_(std::move(rhs.event_))
		, counter_(std::move(rhs.counter_))
	{
		rhs.event_ = nullptr;
		rhs.counter_ = nullptr;
	}

	Event& operator=(const Event& rhs)
	{
		release();
		event_ = rhs.event_;
		counter_ = rhs.counter_;
		assert(counter_);
		++(*counter_);
		return *this;
	}

	Event& operator=(Event&& rhs) noexcept
	{
		release();
		event_ = std::move(rhs.event_);
		counter_ = std::move(rhs.counter_);
		assert(counter_);
		rhs.event_ = nullptr;
		rhs.counter_ = nullptr;
		return *this;
	}

	void swap(Event& rhs) throw()
	{
		std::swap(event_, rhs.event_);
		std::swap(counter_, rhs.counter_);
	}

	~Event()
	{
		release();
	}

	cudaEvent_t event() const { return event_; }

	/**
	 * Records an event into this instance after all preceding operations in <code>stream</code> have been completed.
	 */
	void record(cudaStream_t stream) const
	{
		CUMAT_SAFE_CALL(cudaEventRecord(event_, stream));
	}

	/**
	 * Synchronizes the given stream with this event.
	 * The given stream will wait until the event recorded with \ref record(cudaStream_t) has completed.
	 */
	void streamWait(cudaStream_t stream) const
	{
		CUMAT_SAFE_CALL(cudaStreamWaitEvent(stream, event_, 0));
	}

	/**
	 * Returns the elapsed time in milliseconds between the two recorded events.
	 */
	static float elapsedTime(const Event& start, const Event& end)
	{
		float ms;
		CUMAT_SAFE_CALL(cudaEventSynchronize(start.event()));
		CUMAT_SAFE_CALL(cudaEventSynchronize(end.event()));
		CUMAT_SAFE_CALL(cudaEventElapsedTime(&ms, start.event(), end.event()));
		return ms;
	}
};

CUMAT_NAMESPACE_END

#endif