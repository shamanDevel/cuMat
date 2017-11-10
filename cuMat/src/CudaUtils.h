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
	__device__ CUMAT_STRONG_INLINE const T& load(T* ptr)
	{
#if __CUDA_ARCH__ >= 350
		return __ldg(ptr);
#else
		return *ptr;
#endif
	}
}

CUMAT_NAMESPACE_END

#endif