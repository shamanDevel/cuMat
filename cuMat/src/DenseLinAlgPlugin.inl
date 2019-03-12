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
 * \brief Computes and returns the Cholesky decompositionof this matrix.
 * The matrix must be Hermetian and positive definite.
 * The resulting decomposition can then be used to compute the determinant of the matrix,
 * invert the matrix and solve multiple linear equation systems.
 */
CholeskyDecomposition<_Derived> decompositionCholesky() const
{
    return CholeskyDecomposition<_Derived>(derived());
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

/**
 * \brief Computes the determinant of this matrix.
 * For matrices of up to 4x4, an explicit formula is used. For larger matrices, this method falls back to a Cholesky Decomposition.
 * \return the inverse of this matrix
 */
InverseOp<_Derived> inverse() const
{
	return InverseOp<_Derived>(derived());
}

/** 
  * \brief Computation of matrix inverse and determinant in one kernel call.
  *
  * This is only for fixed-size square matrices of size up to 4x4.
  *
  * \param inverse Reference to the matrix in which to store the inverse.
  * \param determinant Reference to the variable in which to store the determinant.
  *
  * \see inverse(), determinant()
  */
template<typename InverseType, typename DetType>
void computeInverseAndDet(InverseType& inverseOut, DetType& detOut) const
{
	CUMAT_STATIC_ASSERT(Rows >= 1 && Rows <= 4, "This matrix must be a compile-time 1x1, 2x2, 3x3 or 4x4 matrix");
	CUMAT_STATIC_ASSERT(Columns >= 1 && Columns <= 4, "This matrix must be a compile-time 1x1, 2x2, 3x3 or 4x4 matrix");
	CUMAT_STATIC_ASSERT(Rows == Columns, "This matrix must be symmetric");
	CUMAT_STATIC_ASSERT(Rows >= 1 && internal::traits<InverseType>::RowsAtCompileTime, "The output matrix must have the same compile-time size as this matrix");
	CUMAT_STATIC_ASSERT(Columns >= 1 && internal::traits<InverseType>::ColsAtCompileTime, "The output matrix must have the same compile-time size as this matrix");
	CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(Batches > 0 && internal::traits<InverseType>::BatchesAtCompileTime > 0, Batches == internal::traits<InverseType>::BatchesAtCompileTime),
		"This matrix and the output matrix must have the same batch size");
	CUMAT_ASSERT_DIMENSION(batches() == inverseOut.batches());

	CUMAT_STATIC_ASSERT(internal::traits<DetType>::RowsAtCompileTime == 1, "The determinant output must be a (batched) scalar, i.e. compile-time 1x1 matrix");
	CUMAT_STATIC_ASSERT(internal::traits<DetType>::ColsAtCompileTime == 1, "The determinant output must be a (batched) scalar, i.e. compile-time 1x1 matrix");
	CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(Batches > 0 && internal::traits<DetType>::BatchesAtCompileTime > 0, Batches == internal::traits<DetType>::BatchesAtCompileTime),
		"This matrix and the determinant matrix must have the same batch size");
	CUMAT_ASSERT_DIMENSION(batches() == detOut.batches());

	CUMAT_NAMESPACE ComputeInverseWithDet<_Derived, Rows, InverseType, DetType>::run(derived(), inverseOut, detOut);
}
