//Included inside Matrix

////first pull the members from MatrixBase, defined in MatrixBlockPluginRvalue to this scope
////this makes the const versions available here
//using Base::block;
////Apparently, this is not legal C++ code, even if Visual Studio and some versions of GCC allows that (other GCC versions don't)
////So I have to manually provide the const versions

//most general version, static size

/**
* \brief Creates a block of the matrix of static size.
* By using this method, you can convert a dynamically-sized matrix into a statically sized one.
* This is the non-const version that also works as a lvalue reference. Hence, you can overwrite a part of the underlying matrix
* by setting the block to some new expression
*
* \param start_row the start row of the block (zero based)
* \param start_column the start column of the block (zero based)
* \param start_batch the start batch of the block (zero based)
* \tparam NRows the number of rows of the block on compile time
* \tparam NColumsn the number of columns of the block on compile time
* \tparam NBatches the number of batches of the block on compile time
*/
template<int NRows, int NColumns, int NBatches>
MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, Type>
block(Index start_row, Index start_column, Index start_batch, Index num_rows = NRows,
    Index num_columns = NColumns, Index num_batches = NBatches)
{
    CUMAT_ASSERT_ARGUMENT(NRows > 0 ? NRows == num_rows : true);
    CUMAT_ASSERT_ARGUMENT(NColumns > 0 ? NColumns == num_columns : true);
    CUMAT_ASSERT_ARGUMENT(NBatches > 0 ? NBatches == num_batches : true);
    CUMAT_ASSERT_ARGUMENT(num_rows >= 0);
    CUMAT_ASSERT_ARGUMENT(num_columns >= 0);
    CUMAT_ASSERT_ARGUMENT(num_batches >= 0);
    CUMAT_ASSERT_ARGUMENT(start_row >= 0);
    CUMAT_ASSERT_ARGUMENT(start_column >= 0);
    CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
    CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
    CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
    CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
    return MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, Type>(
        *this, num_rows, num_columns, num_batches, start_row, start_column, start_batch);
}

//most general version, dynamic size

/**
* \brief Creates a block of the matrix of dynamic size.
* This is the non-const version that also works as a lvalue reference. Hence, you can overwrite a part of the underlying matrix
* by setting the block to some new expression
*
* \param start_row the start row of the block (zero based)
* \param start_column the start column of the block (zero based)
* \param start_batch the start batch of the block (zero based)
* \param num_rows the number of rows in the block
* \param num_columns the number of columns in the block
* \param num_batches the number of batches in the block
*/
MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, Type>
block(Index start_row, Index start_column, Index start_batch, Index num_rows, Index num_columns, Index num_batches)
{
    CUMAT_ASSERT_ARGUMENT(start_row >= 0);
    CUMAT_ASSERT_ARGUMENT(start_column >= 0);
    CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
    CUMAT_ASSERT_ARGUMENT(num_rows > 0);
    CUMAT_ASSERT_ARGUMENT(num_columns > 0);
    CUMAT_ASSERT_ARGUMENT(num_batches > 0);
    CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
    CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
    CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
    return MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, Type>(
        *this, num_rows, num_columns, num_batches, start_row, start_column, start_batch);
}


//Const versions, taken from MatrixBlockPluginRvalue.h and adopted

/**
 * \brief Creates a block of the matrix of static size.
 * By using this method, you can convert a dynamically-sized matrix into a statically sized one.
 * 
 * \param start_row the start row of the block (zero based)
 * \param start_column the start column of the block (zero based)
 * \param start_batch the start batch of the block (zero based)
 * \tparam NRows the number of rows of the block on compile time
 * \tparam NColumsn the number of columns of the block on compile time
 * \tparam NBatches the number of batches of the block on compile time
 */
template<int NRows, int NColumns, int NBatches>
MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, const Type>
block(Index start_row, Index start_column, Index start_batch, Index num_rows = NRows,
    Index num_columns = NColumns, Index num_batches = NBatches) const
{
    CUMAT_ASSERT_ARGUMENT(NRows > 0 ? NRows == num_rows : true);
    CUMAT_ASSERT_ARGUMENT(NColumns > 0 ? NColumns == num_columns : true);
    CUMAT_ASSERT_ARGUMENT(NBatches > 0 ? NBatches == num_batches : true);
    CUMAT_ASSERT_ARGUMENT(num_rows >= 0);
    CUMAT_ASSERT_ARGUMENT(num_columns >= 0);
    CUMAT_ASSERT_ARGUMENT(num_batches >= 0);
    CUMAT_ASSERT_ARGUMENT(start_row >= 0);
    CUMAT_ASSERT_ARGUMENT(start_column >= 0);
    CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
    CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
    CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
    CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
    return MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, const Type>(
        *this, num_rows, num_columns, num_batches, start_row, start_column, start_batch);
}

//most general version, dynamic size

/**
* \brief Creates a block of the matrix of dynamic size.
*
* \param start_row the start row of the block (zero based)
* \param start_column the start column of the block (zero based)
* \param start_batch the start batch of the block (zero based)
* \param num_rows the number of rows in the block
* \param num_columns the number of columns in the block
* \param num_batches the number of batches in the block
*/
MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, const Type>
block(Index start_row, Index start_column, Index start_batch, Index num_rows, Index num_columns, Index num_batches) const
{
    CUMAT_ASSERT_ARGUMENT(start_row >= 0);
    CUMAT_ASSERT_ARGUMENT(start_column >= 0);
    CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
    CUMAT_ASSERT_ARGUMENT(num_rows > 0);
    CUMAT_ASSERT_ARGUMENT(num_columns > 0);
    CUMAT_ASSERT_ARGUMENT(num_batches > 0);
    CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
    CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
    CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
    return MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, const Type>(
        *this, num_rows, num_columns, num_batches, start_row, start_column, start_batch);
}


// TODO: specializations for batch==1, vectors, slices
