#ifndef __CUMAT_CONTEXT_H__
#define __CUMAT_CONTEXT_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thread>
#include <mutex>
#include <algorithm>

#include "Macros.h"
#include "Errors.h"
#include "Logging.h"
#include "Profiling.h"

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

#define CUMAT_KERNEL_AXIS_LOOP(i, virtual_size, axis) \
	for (CUMAT_NAMESPACE Index i = blockIdx.axis * blockDim.axis + threadIdx.axis; \
		 i < virtual_size.axis; \
		 i += blockDim.axis * gridDim.axis)

#define CUMAT_KERNEL_1D_LOOP(i, virtual_size) \
	CUMAT_KERNEL_AXIS_LOOP(i, virtual_size, x)

#define CUMAT_KERNEL_2D_LOOP(i, j, virtual_size) \
	CUMAT_KERNEL_AXIS_LOOP(j, virtual_size, y) \
	CUMAT_KERNEL_AXIS_LOOP(i, virtual_size, x)

#define CUMAT_KERNEL_3D_LOOP(i, j, k, virtual_size) \
	CUMAT_KERNEL_AXIS_LOOP(k, virtual_size, z) \
	CUMAT_KERNEL_AXIS_LOOP(j, virtual_size, y) \
	CUMAT_KERNEL_AXIS_LOOP(i, virtual_size, x)

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
		static thread_local Context INSTANCE;
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
	 * \brief Returns the kernel launch configurations for a 1D launch.
	 * For details on how to use it, see the documentation of
	 * KernelLaunchConfig.
	 * \param size the size of the problem
	 * \param maxBlockSize the maximal size of the block, must be a power of two
	 * \return the launch configuration
	 */
	KernelLaunchConfig createLaunchConfig1D(unsigned int size, unsigned int maxBlockSize = 1<<15) const
	{
		CUMAT_ASSERT_ARGUMENT(size > 0);
        unsigned int blockSize = std::min(maxBlockSize, 256u);
		//TODO: Very simplistic first version
		//Later improve to read the actual pysical thread count per block and pysical block count
		KernelLaunchConfig cfg = {
			dim3(size, 1, 1),
			dim3(blockSize, 1, 1),
			dim3(CUMAT_DIV_UP(size, blockSize), 1, 1)
		};
		return cfg;
	}

	/**
	* \brief Returns the kernel launch configurations for a 2D launch.
	* For details on how to use it, see the documentation of
	* KernelLaunchConfig.
	* \param sizex the size of the problem along x
	* \param sizey the size of the problem along y
	* \return the launch configuration
	*/
	KernelLaunchConfig createLaunchConfig2D(unsigned int sizex, unsigned int sizey) const
	{
		CUMAT_ASSERT_ARGUMENT(sizex > 0);
		CUMAT_ASSERT_ARGUMENT(sizey > 0);
		//TODO: Very simplistic first version
		//Later improve to read the actual pysical thread count per block and pysical block count
        unsigned int blockSizeX = sizey == 1 ? 256u : 32u; //common case: sizey==0 (no batching)
        unsigned int blockSizeY = sizey == 1 ? 1u : 32u;
		KernelLaunchConfig cfg = {
			dim3(sizex, sizey, 1),
			dim3(blockSizeX, blockSizeY, 1),
			dim3(CUMAT_DIV_UP(sizex, blockSizeX), CUMAT_DIV_UP(sizey, blockSizeY), 1)
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
	* \return the launch configuration
	*/
	KernelLaunchConfig createLaunchConfig3D(unsigned int sizex, unsigned int sizey, unsigned int sizez) const
	{
		CUMAT_ASSERT_ARGUMENT(sizex > 0);
		CUMAT_ASSERT_ARGUMENT(sizey > 0);
		CUMAT_ASSERT_ARGUMENT(sizez > 0);
		//TODO: Very simplistic first version
		//Later improve to read the actual pysical thread count per block and pysical block count
		KernelLaunchConfig cfg = {
			dim3(sizex, sizey, sizez),
			dim3(16, 8, 8),
			dim3(CUMAT_DIV_UP(sizex, 16), CUMAT_DIV_UP(sizey, 8), CUMAT_DIV_UP(sizey, 8))
		};
		return cfg;
	}
};


CUMAT_NAMESPACE_END

#endif