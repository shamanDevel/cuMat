#ifndef __CUMAT_NUM_TRAITS_H__
#define __CUMAT_NUM_TRAITS_H__

#include <thrust/complex.h>
#include <cuComplex.h>

#include "Macros.h"

CUMAT_NAMESPACE_BEGIN

/**
 * \brief The complex type 
 * \tparam _Real the underlying real type
 */
template<typename _Real>
using complex = thrust::complex<_Real>;
//TODO: cuBlas needs cuFloatComplex / cuDoubleComplex

CUMAT_NAMESPACE_END

#endif