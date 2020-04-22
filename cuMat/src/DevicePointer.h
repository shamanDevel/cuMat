#ifndef __CUMAT_DEVICE_POINTER_H__
#define __CUMAT_DEVICE_POINTER_H__

#include <cuda_runtime.h>

#include "Macros.h"
#include "Context.h"

CUMAT_NAMESPACE_BEGIN

template <typename T>
class DevicePointer
{
private:
	T* pointer_;
	int* counter_;
	CUMAT_NAMESPACE Context* context_;
    friend class DevicePointer<typename std::remove_const<T>::type>;

    __host__ __device__
	void release()
	{
#ifndef __CUDA_ARCH__
        //no decrement of the counter in CUDA-code, counter is in host-memory
		assert(CUMAT_IMPLIES(counter_, (*counter_) > 0) && "Attempt to calling release() twice");
		if ((counter_) && (--(*counter_) == 0))
		{
			delete counter_;

			//DEBUG
			if (&Context::current() != context_)
			{
				CUMAT_LOG_WARNING(
					"Freeing memory with a different context than the current context.\n"
					"This will likely crash with an invalid-resource-handle error due to different Cub-Allocators");
			}

			context_->freeDevice(pointer_);
		}
#endif
	}

public:
	DevicePointer(size_t size, CUMAT_NAMESPACE Context& ctx)
		: pointer_(nullptr)
		, counter_(nullptr)
	{
		context_ = &ctx;
		pointer_ = static_cast<T*>(context_->mallocDevice(size * sizeof(T)));
		try {
			counter_ = new int(1);
		}
		catch (...)
		{
			context_->freeDevice(const_cast<typename std::remove_cv<T>::type*>(pointer_));
			throw;
		}
	}
	DevicePointer(size_t size)
		: DevicePointer(size, CUMAT_NAMESPACE Context::current())
	{}

    __host__ __device__
	DevicePointer()
		: pointer_(nullptr)
		, counter_(nullptr)
		, context_(nullptr)
	{}

    __host__ __device__
	DevicePointer(const DevicePointer<T>& rhs)
		: pointer_(rhs.pointer_)
		, counter_(rhs.counter_)
		, context_(rhs.context_)
	{
#ifndef __CUDA_ARCH__
        //no increment of the counter in CUDA-code, counter is in host-memory
		if (counter_) {
			++(*counter_);
		}
#endif
	}

    __host__ __device__
	DevicePointer(DevicePointer<T>&& rhs) noexcept
		: pointer_(std::move(rhs.pointer_))
		, counter_(std::move(rhs.counter_))
		, context_(std::move(rhs.context_))
	{
	    rhs.pointer_ = nullptr;
		rhs.counter_ = nullptr;
		rhs.context_ = nullptr;
	}

    __host__ __device__
	DevicePointer<T>& operator=(const DevicePointer<T>& rhs)
	{
		release();
		pointer_ = rhs.pointer_;
		counter_ = rhs.counter_;
		context_ = rhs.context_;
#ifndef __CUDA_ARCH__
        //no increment of the counter in CUDA-code, counter is in host-memory
		if (counter_) {
			++(*counter_);
		}
#endif
		return *this;
	}

    __host__ __device__
	DevicePointer<T>& operator=(DevicePointer<T>&& rhs) noexcept
	{
		release();
		pointer_ = std::move(rhs.pointer_);
		counter_ = std::move(rhs.counter_);
		context_ = std::move(rhs.context_);
		rhs.pointer_ = nullptr;
		rhs.counter_ = nullptr;
		rhs.context_ = nullptr;
		return *this;
	}

    __host__ __device__
	void swap(DevicePointer<T>& rhs) throw()
	{
		std::swap(pointer_, rhs.pointer_);
		std::swap(counter_, rhs.counter_);
		std::swap(context_, rhs.context_);
	}

    __host__ __device__
	~DevicePointer()
	{
		release();
	}

	__host__ __device__ T* pointer() { return pointer_; }
	__host__ __device__ const T* pointer() const { return pointer_; }

	/**
	 * \brief Returns the current value of the reference counter.
	 * This can be used to determine if this memory is used uniquely
	 * by an object.
	 * \return the current number of references
	 */
	size_t getCounter() const { return counter_ ? *counter_ : 0; }
};

CUMAT_NAMESPACE_END

#endif
