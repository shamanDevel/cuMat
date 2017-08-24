#ifndef __CUMAT_MATRIX_H__
#define __CUMAT_MATRIX_H__

#include "Macros.h"
#include "Constants.h"

CUMAT_NAMESPACE_BEGIN


/**
 * \brief The basic matrix class.
 * It is used to store batched matrices and vectors of
 * compile-time constant size or dynamic size.
 * 
 * For the sake of performance, set as many dimensions to compile-time constants as possible.
 * This allows to choose the best algorithm already during compilation.
 * There is no limit on the size of the compile-time dimensions, since all memory lives in 
 * the GPU memory, not on the stack (as opposed to Eigen).
 * 
 * \tparam _Scalar the scalar type of the matrix
 * \tparam _Rows the number of rows, can be a compile-time constant or \b Dynamic
 * \tparam _Columns the number of columns, can be a compile-time constant or \b Dynamic
 * \tparam _Batches the number of batches, can be a compile-time constant or \b Dynamic
 * \tparam _Flags a combination of flags from the \b Flags enum.
 */
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
class Matrix
{
	
};

CUMAT_NAMESPACE_END

#endif