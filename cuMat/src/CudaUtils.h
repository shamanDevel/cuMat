#ifndef __CUMAT_CUDA_UTILS_H__
#define __CUMAT_CUDA_UTILS_H__

#include "Macros.h"

// SOME CUDA UTILITIES

CUMAT_NAMESPACE_BEGIN

namespace cuda
{
	/**
	 * \brief Loads the given scalar value from the specified address,
	 * possible cached.
	 * \tparam T the type of scalar
	 * \param ptr the pointer to the scalar
	 * \return the value at that adress
	 */
	template<typename T>
	__device__ CUMAT_STRONG_INLINE T load(const T* ptr)
	{
		//#if __CUDA_ARCH__ >= 350
		//		return __ldg(ptr);
		//#else
		return *ptr;
		//#endif
	}
#if __CUDA_ARCH__ >= 350
#define LOAD(T)			\
	template<>			\
	__device__ CUMAT_STRONG_INLINE T load<T>(const T* ptr)	\
	{																\
		return __ldg(ptr);											\
	}
#else
#define LOAD(T)
#endif

	LOAD(char);
	LOAD(short);
	LOAD(int);
	LOAD(long);
	LOAD(long long);
	LOAD(unsigned char);
	LOAD(unsigned short);
	LOAD(unsigned int);
	LOAD(unsigned long);
	LOAD(unsigned long long);
	LOAD(int2);
	LOAD(int4);
	LOAD(uint2);
	LOAD(uint4);
	LOAD(float);
	LOAD(float2);
	LOAD(float4);
	LOAD(double);
	LOAD(double2);

#undef LOAD

}

CUMAT_NAMESPACE_END

#endif