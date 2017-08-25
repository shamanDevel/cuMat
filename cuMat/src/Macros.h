#ifndef __CUMAT_MACROS_H__
#define __CUMAT_MACROS_H__

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

/**
 * \brief Define this macro in a class that should not be copyable or assignable
 * \param TypeName the name of the class
 */
#define CUMAT_DISALLOW_COPY_AND_ASSIGN(TypeName)\
	TypeName(const TypeName&) = delete;      \
    void operator=(const TypeName&) = delete

#endif