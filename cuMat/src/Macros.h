#ifndef __CUMAT_MACROS_H__
#define __CUMAT_MACROS_H__

#include <assert.h>
#include <exception>
#include <stdexcept>
#include <cuda_runtime.h>

/*
 * This file contains global macros and type definitions used everywhere
 */

#ifndef CUMAT_NAMESPACE
/**
 * \brief The namespace of the library
 */
#define CUMAT_NAMESPACE ::cuMat::
#endif

#ifndef CUMAT_NAMESPACE_BEGIN
/**
 * \brief Defines the namespace in which everything of cuMat lives in
 */
#define CUMAT_NAMESPACE_BEGIN namespace cuMat {
#endif

#ifndef CUMAT_NAMESPACE_END
/**
 * \brief Closes the namespace opened with CUMAT_NAMESPACE_BEGIN
 */
#define CUMAT_NAMESPACE_END }
#endif

#ifndef CUMAT_FUNCTION_NAMESPACE_BEGIN
/**
 * \brief Defines the namespace in which overloaed math functions are defined.
 * Examples are 'sin(x)' and 'pow(a,b)'.
 * This defaults to "cuMat::functions", but can be changed if needed.
 */
#define CUMAT_FUNCTION_NAMESPACE_BEGIN namespace cuMat { namespace functions {
#endif

#ifndef CUMAT_FUNCTION_NAMESPACE_END
/**
 * \brief Closes the namespace openeded with CUMAT_FUNCTION_NAMESPACE_BEGIN
 */
#define CUMAT_FUNCTION_NAMESPACE_END }}
#endif

#ifndef CUMAT_FUNCTION_NAMESPACE
/**
 * \brief The namespace in which overloaded math functions are defined.
 */
#define CUMAT_FUNCTION_NAMESPACE ::cuMat::functions::
#endif

#ifdef _MSC_VER
//running under MS Visual Studio -> no thread_local
#define CUMAT_THREAD_LOCAL __declspec( thread )
#else
//C++11 compatible
#define CUMAT_THREAD_LOCAL thread_local
#endif

#ifndef CUMAT_EIGEN_SUPPORT
/**
 * \brief Set CUMAT_EIGEN_INTEROP to 1 to enable Eigen interop.
 * This enables the methods to convert between Eigen matrices and cuMat matrices.
 * Default: 0
 */
#define CUMAT_EIGEN_SUPPORT 0
#endif

#ifdef __CUDACC__
/**
 * If the current source file is compiled with the NVCC as a CUDA source file, 
 * this macro is set to one, else to zero.
 */
#define CUMAT_NVCC 1
#else
#define CUMAT_NVCC 0
#endif

#if CUMAT_NVCC==0
#define CUMAT_ERROR_IF_NO_NVCC(name) {THIS_FUNCTION_REQUIRES_THE_FILE_TO_BE_COMPILED_WITH_NVCC name;}
#else
#define CUMAT_ERROR_IF_NO_NVCC(name)
#endif

/**
 * \brief Define this macro in a class that should not be copyable or assignable
 * \param TypeName the name of the class
 */
#define CUMAT_DISALLOW_COPY_AND_ASSIGN(TypeName)\
	TypeName(const TypeName&) = delete;      \
    void operator=(const TypeName&) = delete

#define CUMAT_STR_DETAIL(x) #x
#define CUMAT_STR(x) CUMAT_STR_DETAIL(x)


/*
 * \brief enable verbose error checking after each kernel launch.
 * This implies a synchronization point after kernel.
 * 
 * By default, this is only enabled in a debug build, but it can be 
 * manually activated by defining <code>CUMAT_VERBOSE_ERROR_CHECKING=1</code>
 */
#ifndef CUMAT_VERBOSE_ERROR_CHECKING
#if defined(_DEBUG) || (!defined(NDEBUG) && !defined(_NDEBUG))
#define CUMAT_VERBOSE_ERROR_CHECKING 1
#else
#define CUMAT_VERBOSE_ERROR_CHECKING 0
#endif
#endif

/*
* \brief enable host assertions.
* The assertion macros (CUMAT_ASSERT* without CUMAT_ASSERT_CUDA)
* will throw an exception if triggered.
*
* By default, this is only enabled in a debug build, but it can be
* manually activated by defining <code>CUMAT_ENABLE_HOST_ASSERTIONS=1</code>
*/
#ifndef CUMAT_ENABLE_HOST_ASSERTIONS
#if defined(_DEBUG) || (!defined(NDEBUG) && !defined(_NDEBUG))
#define CUMAT_ENABLE_HOST_ASSERTIONS 1
#else
#define CUMAT_ENABLE_HOST_ASSERTIONS 0
#endif
#endif

/*
* \brief enable device assertions.
* The assertion macros CUMAT_ASSERT_CUDA
* will throw an exception if triggered.
*
* By default, this is only enabled in a debug build, but it can be
* manually activated by defining <code>CUMAT_ENABLE_DEVICE_ASSERTIONS=1</code>
*/
#ifndef CUMAT_ENABLE_DEVICE_ASSERTIONS
#if defined(_DEBUG) || (!defined(NDEBUG) && !defined(_NDEBUG))
#define CUMAT_ENABLE_DEVICE_ASSERTIONS 1
#else
#define CUMAT_ENABLE_DEVICE_ASSERTIONS 0
#endif
#endif

#if CUMAT_ENABLE_HOST_ASSERTIONS==1

/**
 * \brief Runtime assertion, uses assert()
 * Only use for something that should never happen
 * \param x the expression that must be true
 */
#define CUMAT_ASSERT(x) assert(x)

#define CUMAT_ASSERT_ARGUMENT(x) \
	if (!(x)) throw std::invalid_argument(__FILE__ ":" CUMAT_STR(__LINE__) ": Invalid argument: " #x);
#define CUMAT_ASSERT_BOUNDS(x) \
	if (!(x)) throw std::out_of_range(__FILE__ ":" CUMAT_STR(__LINE__) "Out of bounds: " #x);
#define CUMAT_ASSERT_ERROR(x) \
	if (!(x)) throw std::runtime_error(__FILE__ ":" CUMAT_STR(__LINE__) "Runtime Error: " #x);
#define CUMAT_ASSERT_DIMENSION(x) \
	if (!(x)) throw std::invalid_argument(__FILE__ ":" CUMAT_STR(__LINE__) ": Invalid dimensions: " #x);

#else

#define CUMAT_ASSERT(x)
#define CUMAT_ASSERT_ARGUMENT(x)
#define CUMAT_ASSERT_BOUNDS(x)
#define CUMAT_ASSERT_ERROR(x)
#define CUMAT_ASSERT_DIMENSION(x)

#endif

#if CUMAT_ENABLE_DEVICE_ASSERTIONS==1
/**
 * \brief Assertions in device code (if supported)
 * \param x the expression that must be true
 */
#define CUMAT_ASSERT_CUDA(x) assert(x)
#else
#define CUMAT_ASSERT_CUDA(x)
#endif

#define CUMAT_THROW_INVALID_ARGUMENT(msg) \
	throw std::invalid_argument(__FILE__ ":" CUMAT_STR(__LINE__) ": " msg);
#define CUMAT_THROW_RUNTIME_ERROR(msg) \
	throw std::runtime_error(__FILE__ ":" CUMAT_STR(__LINE__) ": " msg);
#define CUMAT_THROW_OUT_OF_RANGE(msg) \
	throw std::out_of_range(__FILE__ ":" CUMAT_STR(__LINE__) ": " msg);

/**
 * \brief A static assertion
 * \param exp the compile-time boolean expression that must be true
 * \param msg an error message if exp is false
 */
#define CUMAT_STATIC_ASSERT(exp, msg) static_assert(exp, msg)

#define CUMAT_STRONG_INLINE __inline__


/**
 * \brief Returns the integer division x/y rounded up.
 * Taken from https://stackoverflow.com/a/2745086/4053176
 */
#define CUMAT_DIV_UP(x, y) (((x) + (y) - 1) / (y))

/**
 * \brief Computes the logical implication (a -> b)
 */
#define CUMAT_IMPLIES(a,b) (!(a) || (b))


#define CUMAT_PUBLIC_API_NO_METHODS                                 \
    enum                                                            \
    {                                                               \
    Flags = internal::traits<Type>::Flags,                          \
    Rows = internal::traits<Type>::RowsAtCompileTime,               \
    Columns = internal::traits<Type>::ColsAtCompileTime,            \
    Batches = internal::traits<Type>::BatchesAtCompileTime          \
    };                                                              \
    using Scalar = typename internal::traits<Type>::Scalar;         \
    using SrcTag = typename internal::traits<Type>::SrcTag;         \
    using DstTag = typename internal::traits<Type>::DstTag;
/**
 * \brief Defines the basic API of each cumat class.
 * The typedefs and enums that have to be exposed.
 * But before that, you have to define the current class in \c Type
 * and the base class in \c Base.
 */
#define CUMAT_PUBLIC_API                                            \
    CUMAT_PUBLIC_API_NO_METHODS                                     \
    using Base::derived;                                            \
    using Base::eval_t;

#endif