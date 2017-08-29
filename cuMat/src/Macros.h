#ifndef __CUMAT_MACROS_H__
#define __CUMAT_MACROS_H__

#include <assert.h>
#include <exception>

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

/**
 * \brief Define this macro in a class that should not be copyable or assignable
 * \param TypeName the name of the class
 */
#define CUMAT_DISALLOW_COPY_AND_ASSIGN(TypeName)\
	TypeName(const TypeName&) = delete;      \
    void operator=(const TypeName&) = delete


//TODO: move assertions to Errors.h ?

/**
 * \brief Runtime assertion, uses assert()
 * Only use for something that should never happen
 * \param x the expression that must be true
 */
#define CUMAT_ASSERT(x) assert(x)


#define CUMAT_ASSERT_ARGUMENT(x) \
	if (!(x)) throw std::invalid_argument("Invalid argument: " #x);
#define CUMAT_ASSERT_BOUNDS(x) \
	if (!(x)) throw std::out_of_range("Out of bounds: " #x);
#define CUMAT_ASSERT_ERROR(x) \
	if (!(x)) throw std::runtime_error("Runtime Error: " #x);

/**
 * \brief Assertions in device code (if supported)
 * \param x the expression that must be true
 */
#define CUMAT_ASSERT_CUDA(x) //do nothing, not supported

#define CUMAT_STRONG_INLINE inline


/**
 * \brief Returns the integer division x/y rounded up.
 * Taken from https://stackoverflow.com/a/2745086/4053176
 */
#define CUMAT_DIV_UP(x, y) (((x) + (y) - 1) / (y))


/**
 * \brief The datatype used for matrix indexing
 */
typedef ptrdiff_t Index;

#endif