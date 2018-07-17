#ifndef __CUMAT_CONSTANTS_H__
#define __CUMAT_CONSTANTS_H__


#include "Macros.h"

CUMAT_NAMESPACE_BEGIN

/** 
 * \brief This value means that a positive quantity (e.g., a size) is not known at compile-time, 
 * and that instead the value is stored in some runtime variable.
 */
const int Dynamic = -1;

/**
 * \brief The bit flags for the matrix expressions.
 */
enum Flags
{
	/**
	 * \brief The storage is column major (the default).
	 */
	ColumnMajor = 0x00,
	/**
	 * \brief The storage is row major.
	 */
	RowMajor = 0x01,

};
#define CUMAT_IS_COLUMN_MAJOR(flags) (((flags) & CUMAT_NAMESPACE Flags::RowMajor)==0)
#define CUMAT_IS_ROW_MAJOR(flags) ((flags) & CUMAT_NAMESPACE Flags::RowMajor)

/**
 * \brief Flags that specify how the data in a MatrixBase-expression can be accessed.
 */
enum AccessFlags
{
    /**
     * \brief component-wise read is available.
     * The following method must be provided:
     * \code
     * __device__ const Scalar& coeff(Index row, Index col, Index batch) const;
     * \endcode
     */
    ReadCwise = 0x01,
    /**
     * \brief direct read is available, the underlying memory is directly adressable.
     * The following method must be provided:
     * \code
     * __host__ __device__ const _Scalar* data() const;
     * __host__ bool isExclusiveUse() const;
     * \endcode
     */
    ReadDirect = 0x02,
    /**
     * \brief Component-wise read is available. 
     * To allow the implementation to specify the access order, the following methods have to be provided:
     * \code
     * __host__ Index size() const;
     * __host__ __device__ void index(Index index, Index& row, Index& col, Index& batch) const;
     *  __device__ void setRawCoeff(Index index, const Scalar& newValue);
     * \endcode
     */
    WriteCwise = 0x10,
    /**
     * \brief Direct write is available, the underlying memory can be directly written to.
     * The following method must be provided:
     * \code
     * __host__ __device__ _Scalar* data()
     * \endcode
     */
    WriteDirect = 0x20,
    /**
     * \brief This extends \c WriteCwise and allows inplace modifications (compound operators) by additionally providing the function
     * \code __device__ const Scalar& getRawCoeff(Index index) const; \endcode .
     * To enable compound assignment with this as target type, either RWCwise or RWCwiseRef (or both) have to be defined.
     */
    RWCwise = 0x40,
    /**
    * \brief This extends \c WriteCwise and allows inplace modifications (compound operators) by additionally providing the function
    * \code __device__ Scalar& rawCoeff(Index index); \endcode for read-write access to that entry.
    * To enable compound assignment with this as target type, either RWCwise or RWCwiseRef (or both) have to be defined.
    */
    RWCwiseRef = 0x80,
};

/**
 * \brief The axis over which reductions are performed.
 */
enum ReductionAxis
{
    Row = 1,
    Column = 2,
    Batch = 4,
    All = Row | Column | Batch
};

/**
* \brief Specifies the assignment mode in \c evalTo() .
* This is the difference between regular assignment (operator==, \c AssignmentMode::ASSIGN)
* and inplace modifications like operator+= (\c AssignmentMode::ADD).
*
* Note that not all assignment modes have to be supported for all scalar types
* and all right hand sides.
* For example:
*  - MUL (=*) and DIV (=\) are only supported for scalar right hand sides (broadcasting)
*    to avoid the ambiguity if component-wise or matrix operations are meant
*  - MOD (%=), AND (&=), OR (|=) are only supported for integer types
*/
enum class AssignmentMode
{
    ASSIGN,
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    AND,
    OR,
};

/**
 * \brief Flags for the SparseMatrix class.
 */
enum SparseFlags
{
    /**
     * \brief Matrix stored in the Compressed Sparse Column format.
     */
    CSC = 1,
    /**
    * \brief Matrix stored in the Compressed Sparse Row format.
    */
    CSR = 2
};

CUMAT_NAMESPACE_END

#endif