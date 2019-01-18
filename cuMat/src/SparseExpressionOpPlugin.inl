//Included in MatrixBase

/**
 * \brief Views this matrix expression as a sparse matrix.
 * This enforces the specified sparsity pattern and the coefficients
 * of this matrix expression are then only evaluated at these positions.
 * 
 * For now, the only use case is the sparse matrix-vector product.
 * For example:
 * \code
 * SparseMatrix<...> m1, m2; //all initialized with the same sparsity pattern
 * VectorXf v1, v2 = ...;
 * v1 = (m1 + m2) * v2;
 * \endcode
 * In this form, this would trigger a dense matrix vector multiplication, which is 
 * completely unfeasable. This is because the the addition expression 
 * does not know anything about sparse matrices and the product operation
 * then only sees an addition expression on the left hand side. Thus,
 * because of lacking knowledge, it has to trigger a dense evaluation.
 * 
 * Improvement:
 * \code
 * v1 = (m1 + m2).sparseView<Format>(m1.getSparsityPattern()) * v2;
 * \endcode
 * with Format being either CSR or CSC.
 * This enforces the sparsity pattern of m1 onto the addition expression.
 * Thus the (immediately following!) product expression sees this sparse expression
 * and can trigger a sparse matrix-vector multiplication.
 * But the sparse matrices \c m1 and \c m2 now have to search for their coefficients.
 * This is because they don't know that the parent operations (add+multiply) will call
 * their coefficients in order of their own entries. In other words, that the \c linear
 * index parameter in \ref coeff(Index row, Index col, Index batch, Index linear) matches
 * the linear index in their data array.
 * (This is a valid assumption if you take transpose and block operations that change the
 * access pattern into considerations. Further, this allows e.g. \c m2 to have a different
 * sparsity pattern from m1, but only the entries that are included in both are used.)
 * 
 * To overcome the above problem, one has to make one last adjustion:
 * \code
 * v1 = (m1.direct() + m2.direct()).sparseView<Format>(m1.getSparsityPattern()) * v2;
 * \endcode
 * \ref SparseMatrix::direct() tells the matrix that the linear index in 
 * \ref coeff(Index row, Index col, Index batch, Index linear) matches the linear index
 * in the data array and thus can be used directly. This discards and checks that 
 * the row, column and batch index actually match. So use this with care
 * if you know that access pattern is not changed in the operation.
 * (This holds true for all non-broadcasting component wise expressions)
 * 
 * \param pattern the enforced sparsity pattern
 * \tparam _SparseFlags the sparse format: CSC or CSR
 */
template<SparseFlags _SparseFlags>
SparseExpressionOp<Type, _SparseFlags>
sparseView(const SparsityPattern<_SparseFlags>& pattern)
{
    return SparseExpressionOp<Type, _SparseFlags>(derived(), pattern);
}