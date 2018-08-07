#ifndef __CUMAT_NUM_TRAITS_H__
#define __CUMAT_NUM_TRAITS_H__

#include <cuComplex.h>

#include "Macros.h"
#include "ForwardDeclarations.h"
#include <type_traits>
#include <limits>

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
        static constexpr CUMAT_STRONG_INLINE RealType epsilon() {return std::numeric_limits<T>::epsilon();}
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
        static constexpr CUMAT_STRONG_INLINE RealType epsilon() {return std::numeric_limits<float>::epsilon();}
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
        static constexpr CUMAT_STRONG_INLINE RealType epsilon() {return std::numeric_limits<double>::epsilon();}
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
        static constexpr CUMAT_STRONG_INLINE RealType epsilon() {return std::numeric_limits<float>::epsilon();}
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
        static constexpr CUMAT_STRONG_INLINE RealType epsilon() {return std::numeric_limits<double>::epsilon();}
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

    template<typename T>
    struct NumOps //special functions for numbers
    {
        static __host__ __device__ CUMAT_STRONG_INLINE T conj(const T& v) {return v;}
    };
    template<>
    struct NumOps<cfloat>
    {
        static __host__ __device__ CUMAT_STRONG_INLINE cfloat conj(const cfloat& v) {return ::thrust::conj(v);}
    };
    template<>
    struct NumOps<cdouble>
    {
        static __host__ __device__ CUMAT_STRONG_INLINE cdouble conj(const cdouble& v) {return ::thrust::conj(v);}
    };
}

CUMAT_NAMESPACE_END

#endif