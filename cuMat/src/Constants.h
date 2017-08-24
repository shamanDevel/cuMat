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
	ColumnMajorBit = 0x00,
	/**
	 * \brief The storage is row major.
	 */
	RowMajorBit = 0x01,

};

CUMAT_NAMESPACE_END

#endif