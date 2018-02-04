//Included inside MatrixBase, define the accessors

/**
 * \brief Computes and returns the LU decomposition with pivoting of this matrix.
 * The resulting decomposition can then be used to compute the determinant of the matrix,
 * invert the matrix and solve multiple linear equation systems.
 */
LUDecomposition<_Derived> decompositionLU() const
{
    return LUDecomposition<_Derived>(derived());
}

/**
 * \brief Computes the determinant of this matrix.
 * \return the determinant of this matrix
 */
DeterminantOp<_Derived> determinant() const
{
    return DeterminantOp<_Derived>(derived());
}

/**
* \brief Computes the log-determinant of this matrix.
* This is only supported for hermitian positive definite matrices, because no sign is computed.
* A negative determinant would return in an complex logarithm which requires to return
* a complex result for real matrices. This is not desired.
* \return the log-determinant of this matrix
*/
Matrix<typename internal::traits<_Derived>::Scalar, 1, 1, internal::traits<_Derived>::BatchesAtCompileTime, ColumnMajor> logDeterminant() const
{
    //TODO: implement direct methods for matrices up to 4x4.
    return decompositionLU().logDeterminant();
}


