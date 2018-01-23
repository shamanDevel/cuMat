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
        enum
        {
            IsCudaNumeric = 0,
            IsComplex = false
        };
	};

    template <>
    struct NumTraits<float>
    {
        typedef float Type;
        typedef float RealType;
        enum
        {
            IsCudaNumeric = 1,
            IsComplex = false,
        };
    };
    template <>
    struct NumTraits<double>
    {
        typedef double Type;
        typedef double RealType;
        enum
        {
            IsCudaNumeric = 1,
            IsComplex = false,
        };
    };

	template <>
	struct NumTraits<cfloat>
	{
		typedef cfloat Type;
		typedef float RealType;
        enum
        {
            IsCudaNumeric = 1,
            IsComplex = true,
        };
	};

	template <>
	struct NumTraits<cdouble>
	{
		typedef cdouble Type;
		typedef double RealType;
        enum
        {
            IsCudaNumeric = 1,
            IsComplex = true,
        };
	};
}

CUMAT_NAMESPACE_END

#endif