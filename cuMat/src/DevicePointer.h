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
	size_t* counter_;
	cuMat::Context* context_;

	void release()
	{
		if ((counter_) && (--(*counter_) == 0))
		{
			delete counter_;
			context_->freeDevice(pointer_);
		}
	}

public:
	DevicePointer(size_t size) 
		: pointer_(nullptr)
		, counter_(nullptr)
	{
		context_ = &cuMat::Context::current();
		pointer_ = static_cast<T*>(context_->mallocDevice(size * sizeof(T)));
		try {
			counter_ = new size_t(1);
		}
		catch (...)
		{
			context_->freeDevice(pointer_);
			throw;
		}
	}

	DevicePointer()
		: pointer_(nullptr)
		, counter_(nullptr)
		, context_(nullptr)
	{}

	DevicePointer(const DevicePointer<T>& rhs)
		: pointer_(rhs.pointer_)
		, counter_(rhs.counter_)
		, context_(rhs.context_)
	{
		if (counter_) {
			++(*counter_);
		}
	}

	DevicePointer(DevicePointer<T>&& rhs) noexcept
		: pointer_(std::move(rhs.pointer_))
		, counter_(std::move(rhs.counter_))
		, context_(std::move(rhs.context_))
	{}

	DevicePointer<T>& operator=(const DevicePointer<T>& rhs)
	{
		release();
		pointer_ = rhs.pointer_;
		counter_ = rhs.counter_;
		context_ = rhs.context_;
		if (counter_) {
			++(*counter_);
		}
		return *this;
	}

	DevicePointer<T>& operator=(DevicePointer<T>&& rhs) noexcept
	{
		release();
		pointer_ = std::move(rhs.pointer_);
		counter_ = std::move(rhs.counter_);
		context_ = std::move(rhs.context_);
		return *this;
	}

	void swap(DevicePointer<T>& rhs) throw()
	{
		std::swap(pointer_, rhs.pointer_);
		std::swap(counter_, rhs.counter_);
		std::swap(context_, rhs.context_);
	}

	~DevicePointer()
	{
		release();
	}

	__host__ __device__ T* pointer() { return pointer_; }
	__host__ __device__ const T* pointer() const { return pointer_; }
};

CUMAT_NAMESPACE_END

#endif