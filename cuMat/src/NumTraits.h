#ifndef __CUMAT_NUM_TRAITS_H__
#define __CUMAT_NUM_TRAITS_H__

#include <cuComplex.h>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include <type_traits>

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

    template <typename T>
    struct isPrimitive : std::is_arithmetic<T> {};
    template <> struct isPrimitive<cfloat> : std::integral_constant<bool, true> {};
    template <> struct isPrimitive<cdouble> : std::integral_constant<bool, true> {};

    /**
     * \brief Can the type T be used for broadcasting when the scalar type of the other matrix is S?
     */
    template<typename T, typename S>
    struct canBroadcast : std::integral_constant<bool,
        (std::is_convertible<T, S>::value && CUMAT_NAMESPACE internal::isPrimitive<T>::value)  \
        || std::is_same<typename std::remove_cv<T>::type, typename std::remove_cv<S>::type>::value
    > {};
}

CUMAT_NAMESPACE_END

#endif