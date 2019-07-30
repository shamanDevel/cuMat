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
	CUMAT_ERROR_IF_NO_NVCC(block)
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
	CUMAT_ERROR_IF_NO_NVCC(block)
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
	CUMAT_ERROR_IF_NO_NVCC(block)
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
	CUMAT_ERROR_IF_NO_NVCC(block)
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


/**
* \brief Extracts a row out of the matrix.
* \param row the index of the row
*/
MatrixBlock<_Scalar, 1, _Columns, _Batches, _Flags, const Type>
row(Index row) const
{
	CUMAT_ERROR_IF_NO_NVCC(row)
    CUMAT_ASSERT_ARGUMENT(row >= 0);
    CUMAT_ASSERT_ARGUMENT(row < rows());
    return MatrixBlock<_Scalar, 1, _Columns, _Batches, _Flags, const Type>(
            *this, 1, cols(), batches(), row, 0, 0);
}
/**
* \brief Extracts a row out of the matrix.
* \param row the index of the row
*/
MatrixBlock<_Scalar, 1, _Columns, _Batches, _Flags, Type>
row(Index row)
{
	CUMAT_ERROR_IF_NO_NVCC(row)
    CUMAT_ASSERT_ARGUMENT(row >= 0);
    CUMAT_ASSERT_ARGUMENT(row < rows());
    return MatrixBlock<_Scalar, 1, _Columns, _Batches, _Flags, Type>(
        *this, 1, cols(), batches(), row, 0, 0);
}

/**
* \brief Extracts a column out of the matrix.
* \param col the index of the column
*/
MatrixBlock<_Scalar, _Rows, 1, _Batches, _Flags, const Type>
col(Index column) const
{
	CUMAT_ERROR_IF_NO_NVCC(col)
    CUMAT_ASSERT_ARGUMENT(column >= 0);
    CUMAT_ASSERT_ARGUMENT(column < cols());
    return MatrixBlock<_Scalar, _Rows, 1, _Batches, _Flags, const Type>(
            *this, rows(), 1, batches(), 0, column, 0);
}

/**
* \brief Extracts a column out of the matrix.
* \param col the index of the column
*/
MatrixBlock<_Scalar, _Rows, 1, _Batches, _Flags, Type>
col(Index column)
{
	CUMAT_ERROR_IF_NO_NVCC(col)
    CUMAT_ASSERT_ARGUMENT(column >= 0);
    CUMAT_ASSERT_ARGUMENT(column < cols());
    return MatrixBlock<_Scalar, _Rows, 1, _Batches, _Flags, Type>(
        *this, rows(), 1, batches(), 0, column, 0);
}

/**
* \brief Extracts a slice of a specific batch out of the batched matrix
* \param batch the index of the batch
*/
MatrixBlock<_Scalar, _Rows, _Columns, 1, _Flags, const Type>
slice(Index batch) const
{
	CUMAT_ERROR_IF_NO_NVCC(slice)
    CUMAT_ASSERT_ARGUMENT(batch >= 0);
    CUMAT_ASSERT_ARGUMENT(batch < batches());
    return MatrixBlock<_Scalar, _Rows, _Columns, 1, _Flags, const Type>(
            *this, rows(), cols(), 1, 0, 0, batch);
}
/**
* \brief Extracts a slice of a specific batch out of the batched matrix
* \param batch the index of the batch
*/
MatrixBlock<_Scalar, _Rows, _Columns, 1, _Flags, Type>
slice(Index batch)
{
	CUMAT_ERROR_IF_NO_NVCC(slice)
    CUMAT_ASSERT_ARGUMENT(batch >= 0);
    CUMAT_ASSERT_ARGUMENT(batch < batches());
    return MatrixBlock<_Scalar, _Rows, _Columns, 1, _Flags, Type>(
        *this, rows(), cols(), 1, 0, 0, batch);
}

// Vector operations

private:
template<int N>
MatrixBlock<_Scalar, N, 1, _Batches, _Flags, const Type>
segmentHelper(Index start, std::true_type) const
{
    //column vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + N <= rows());
    return MatrixBlock<_Scalar, N, 1, _Batches, _Flags, const Type>(
            *this, N, 1, batches(), start, 0, 0);
}
template<int N>
MatrixBlock<_Scalar, N, 1, _Batches, _Flags, Type>
segmentHelper(Index start, std::true_type)
{
    //column vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + N <= rows());
    return MatrixBlock<_Scalar, N, 1, _Batches, _Flags, Type>(
        *this, N, 1, batches(), start, 0, 0);
}

template<int N>
MatrixBlock< _Scalar, 1, N, _Batches, _Flags, const Type>
segmentHelper(Index start, std::false_type) const
{
    //row vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + N <= cols());
    return MatrixBlock<_Scalar, 1, N, _Batches, _Flags, const Type>(
            *this, 1, N, batches(), 0, start, 0);
}
template<int N>
MatrixBlock< _Scalar, 1, N, _Batches, _Flags, Type>
segmentHelper(Index start, std::false_type)
{
    //row vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + N <= cols());
    return MatrixBlock<_Scalar, 1, N, _Batches, _Flags, Type>(
        *this, 1, N, batches(), 0, start, 0);
}

public:

/**
* \brief Extracts a fixed-size segment of the vector.
*   Only available for vectors
* \param start the start position of the segment
* \tparam N the length of the segment
*/
template<int N>
auto //FixedVectorSegmentXpr<N>::Type
segment(Index start) const -> decltype(segmentHelper<N>(start, std::integral_constant<bool, _Columns == 1>()))
{
	CUMAT_ERROR_IF_NO_NVCC(segment)
    CUMAT_STATIC_ASSERT(
        (_Rows == 1 || _Columns == 1),
        "segment can only act on compile-time vectors");
    return segmentHelper<N>(start, std::integral_constant<bool, _Columns == 1>());
}

/**
* \brief Extracts a fixed-size segment of the vector.
*   Only available for vectors.
*   Non-const version, allows to modify the vector.
* \param start the start position of the segment
* \tparam N the length of the segment
*/
template<int N>
auto //FixedVectorSegmentXpr<N>::Type
segment(Index start) -> decltype(segmentHelper<N>(start, std::integral_constant<bool, _Columns == 1>()))
{
	CUMAT_ERROR_IF_NO_NVCC(segment)
    CUMAT_STATIC_ASSERT(
        (_Rows == 1 || _Columns == 1),
        "segment can only act on compile-time vectors");
    return segmentHelper<N>(start, std::integral_constant<bool, _Columns == 1>());
}

/**
 * \brief Extracts a fixed-size segment from the head of the vector.
 * Only available for vectors
 * \tparam N the length of the segment
 */
template<int N>
auto head() const -> decltype(segment<N>(0))
{
	CUMAT_ERROR_IF_NO_NVCC(head)
    return segment<N>(0);
}
/**
* \brief Extracts a fixed-size segment from the head of the vector.
* Only available for vectors.
* Non-const version
* \tparam N the length of the segment
*/
template<int N>
auto head() -> decltype(segment<N>(0))
{
	CUMAT_ERROR_IF_NO_NVCC(head)
    return segment<N>(0);
}

/**
* \brief Extracts a fixed-size segment from the tail of the vector.
* Only available for vectors
* \tparam N the length of the segment
*/
template<int N>
auto tail() const -> decltype(segment<N>(0))
{
	CUMAT_ERROR_IF_NO_NVCC(tail)
    return segment<N>(std::max(rows(), cols()) - N);
}
/**
* \brief Extracts a fixed-size segment from the tail of the vector.
* Only available for vectors.
* Non-const version
* \tparam N the length of the segment
*/
template<int N>
auto tail() -> decltype(segment<N>(0))
{
	CUMAT_ERROR_IF_NO_NVCC(tail)
    return segment<N>(std::max(rows(), cols()) - N);
}

private:
MatrixBlock<_Scalar, Dynamic, 1, _Batches, _Flags, const Type>
segmentHelper(Index start, Index length, std::true_type) const
{
    //column vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + length <= rows());
    return MatrixBlock<_Scalar, Dynamic, 1, _Batches, _Flags, const Type>(
            *this, length, 1, batches(), start, 0, 0);
}
MatrixBlock<_Scalar, Dynamic, 1, _Batches, _Flags, Type>
segmentHelper(Index start, Index length, std::true_type)
{
    //column vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + length <= rows());
    return MatrixBlock<_Scalar, Dynamic, 1, _Batches, _Flags, Type>(
        *this, length, 1, batches(), start, 0, 0);
}
MatrixBlock<_Scalar, 1, Dynamic, _Batches, _Flags, const Type>
segmentHelper(Index start, Index length, std::false_type) const
{
    //row vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + length <= cols());
    return MatrixBlock<_Scalar, 1, Dynamic, _Batches, _Flags, const Type>(
            *this, 1, length, batches(), 0, start, 0);
}
MatrixBlock<_Scalar, 1, Dynamic, _Batches, _Flags, Type>
segmentHelper(Index start, Index length, std::false_type)
{
    //row vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + length <= cols());
    return MatrixBlock<_Scalar, 1, Dynamic, _Batches, _Flags, Type>(
        *this, 1, length, batches(), 0, start, 0);
}

public:
/**
* \brief Extracts a dynamic-size segment of the vector.
*   Only available for vectors
* \param start the start position of the segment
* \param length the length of the segment
*/
auto
segment(Index start, Index length) const -> decltype(segmentHelper(start, length, std::integral_constant<bool, _Columns == 1>()))
{
	CUMAT_ERROR_IF_NO_NVCC(segment)
    CUMAT_STATIC_ASSERT(
        (_Rows == 1 || _Columns == 1),
        "segment can only act on compile-time vectors");
    return segmentHelper(start, length, std::integral_constant<bool, _Columns == 1>());
}
/**
* \brief Extracts a dynamic-size segment of the vector.
*   Only available for vectors.
*  Non-const version
* \param start the start position of the segment
* \param length the length of the segment
*/
auto
segment(Index start, Index length) -> decltype(segmentHelper(start, length, std::integral_constant<bool, _Columns == 1>()))
{
	CUMAT_ERROR_IF_NO_NVCC(segment)
    CUMAT_STATIC_ASSERT(
        (_Rows == 1 || _Columns == 1),
        "segment can only act on compile-time vectors");
    return segmentHelper(start, length, std::integral_constant<bool, _Columns == 1>());
}

/**
* \brief Extracts a dynamic-size segment from the head of the vector.
* Only available for vectors
* \param length the length of the segment
*/
auto head(Index length) const -> decltype(segment(0, length))
{
	CUMAT_ERROR_IF_NO_NVCC(head)
    return segment(0, length);
}
/**
* \brief Extracts a dynamic-size segment from the head of the vector.
* Only available for vectors.
* Non-const version.
* \param length the length of the segment
*/
auto head(Index length) -> decltype(segment(0, length))
{
	CUMAT_ERROR_IF_NO_NVCC(head)
    return segment(0, length);
}

/**
* \brief Extracts a dynamic-size segment from the tail of the vector.
* Only available for vectors
* \param length the length of the segment
*/
auto tail(Index length) const -> decltype(segment(0, length))
{
	CUMAT_ERROR_IF_NO_NVCC(tail)
    return segment(std::max(rows(), cols()) - length, length);
}
/**
* \brief Extracts a dynamic-size segment from the tail of the vector.
* Only available for vectors.
* Non-const version
* \param length the length of the segment
*/
auto tail(Index length) -> decltype(segment(0, length))
{
	CUMAT_ERROR_IF_NO_NVCC(tail)
    return segment(std::max(rows(), cols()) - length, length);
}
