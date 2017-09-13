#ifndef __CUMAT_NUM_TRAITS_H__
#define __CUMAT_NUM_TRAITS_H__

#include <cuComplex.h>

#include "Macros.h"
#include "ForwardDeclarations.h"

CUMAT_NAMESPACE_BEGIN

namespace internal 
{
	template <typename T>
	struct NumTraits
	{
		typedef T Type;
		typedef T RealType;
	};

	template <>
	struct NumTraits<cfloat>
	{
		typedef cfloat Type;
		typedef float RealType;
	};

	template <>
	struct NumTraits<cdouble>
	{
		typedef cdouble Type;
		typedef double RealType;
	};
}

CUMAT_NAMESPACE_END

#endif