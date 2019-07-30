//Included inside MatrixBase

//most general version, static size

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
MatrixBlock<typename internal::traits<_Derived>::Scalar, NRows, NColumns, NBatches, internal::traits<_Derived>::Flags, const _Derived>
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
    return MatrixBlock<typename internal::traits<_Derived>::Scalar, NRows, NColumns, NBatches, internal::traits<_Derived>::Flags, const _Derived>(
        derived(), num_rows, num_columns, num_batches, start_row, start_column, start_batch);
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
MatrixBlock<typename internal::traits<_Derived>::Scalar, Dynamic, Dynamic, Dynamic, internal::traits<_Derived>::Flags, const _Derived>
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
    return MatrixBlock<typename internal::traits<_Derived>::Scalar, Dynamic, Dynamic, Dynamic, internal::traits<_Derived>::Flags, const _Derived>(
        derived(), num_rows, num_columns, num_batches, start_row, start_column, start_batch);
}

// specializations for batch==1, vectors, slices

/**
* \brief Extracts a row out of the matrix.
* \param row the index of the row
*/
MatrixBlock<
    typename internal::traits<_Derived>::Scalar, 
    1, internal::traits<_Derived>::ColsAtCompileTime, internal::traits<_Derived>::BatchesAtCompileTime,
    internal::traits<_Derived>::Flags, const _Derived>
row(Index row) const
{
	CUMAT_ERROR_IF_NO_NVCC(row)
    CUMAT_ASSERT_ARGUMENT(row >= 0);
    CUMAT_ASSERT_ARGUMENT(row < rows());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        1, internal::traits<_Derived>::ColsAtCompileTime, internal::traits<_Derived>::BatchesAtCompileTime,
        internal::traits<_Derived>::Flags, const _Derived>(
        derived(), 1, cols(), batches(), row, 0, 0);
}

/**
* \brief Extracts a column out of the matrix.
* \param col the index of the column
*/
MatrixBlock<
    typename internal::traits<_Derived>::Scalar,
    internal::traits<_Derived>::RowsAtCompileTime, 1, internal::traits<_Derived>::BatchesAtCompileTime,
    internal::traits<_Derived>::Flags, const _Derived>
col(Index column) const
{
	CUMAT_ERROR_IF_NO_NVCC(col)
    CUMAT_ASSERT_ARGUMENT(column >= 0);
    CUMAT_ASSERT_ARGUMENT(column < cols());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        internal::traits<_Derived>::RowsAtCompileTime, 1, internal::traits<_Derived>::BatchesAtCompileTime,
        internal::traits<_Derived>::Flags, const _Derived>(
            derived(), rows(), 1, batches(), 0, column, 0);
}

/**
* \brief Extracts a slice of a specific batch out of the batched matrix
* \param batch the index of the batch
*/
MatrixBlock<
    typename internal::traits<_Derived>::Scalar,
    internal::traits<_Derived>::RowsAtCompileTime, internal::traits<_Derived>::ColsAtCompileTime, 1,
    internal::traits<_Derived>::Flags, const _Derived>
slice(Index batch) const
{
	CUMAT_ERROR_IF_NO_NVCC(slice)
    CUMAT_ASSERT_ARGUMENT(batch >= 0);
    CUMAT_ASSERT_ARGUMENT(batch < batches());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        internal::traits<_Derived>::RowsAtCompileTime, internal::traits<_Derived>::ColsAtCompileTime, 1,
        internal::traits<_Derived>::Flags, const _Derived>(
            derived(), rows(), cols(), 1, 0, 0, batch);
}


// Vector operations

private:
template<int N>
MatrixBlock<
    typename internal::traits<_Derived>::Scalar,
    N,
    1,
    internal::traits<_Derived>::BatchesAtCompileTime,
    internal::traits<_Derived>::Flags, const _Derived>
segmentHelper(Index start, std::true_type) const
{
    //column vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + N <= rows());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        N, 1, internal::traits<_Derived>::BatchesAtCompileTime,
        internal::traits<_Derived>::Flags, const _Derived>(
            derived(), N, 1, batches(), start, 0, 0);
}

template<int N>
MatrixBlock<
    typename internal::traits<_Derived>::Scalar,
    1,
    N,
    internal::traits<_Derived>::BatchesAtCompileTime,
    internal::traits<_Derived>::Flags, const _Derived>
segmentHelper(Index start, std::false_type) const
{
    //row vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + N <= cols());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        1, N, internal::traits<_Derived>::BatchesAtCompileTime,
        internal::traits<_Derived>::Flags, const _Derived>(
            derived(), 1, N, batches(), 0, start, 0);
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
segment(Index start) const -> decltype(segmentHelper<N>(start, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>()))
{
	CUMAT_ERROR_IF_NO_NVCC(segment)
    CUMAT_STATIC_ASSERT(
        (internal::traits<_Derived>::RowsAtCompileTime == 1 || internal::traits<_Derived>::ColsAtCompileTime == 1),
        "segment can only act on compile-time vectors");
    return segmentHelper<N>(start, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>());
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

private:
MatrixBlock<
    typename internal::traits<_Derived>::Scalar,
    Dynamic,
    1,
    internal::traits<_Derived>::BatchesAtCompileTime,
    internal::traits<_Derived>::Flags, const _Derived>
segmentHelper(Index start, Index length, std::true_type) const
{
    //column vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + length <= rows());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        Dynamic, 1, internal::traits<_Derived>::BatchesAtCompileTime,
        internal::traits<_Derived>::Flags, const _Derived>(
            derived(), length, 1, batches(), start, 0, 0);
}
MatrixBlock<
    typename internal::traits<_Derived>::Scalar,
    1,
    Dynamic,
    internal::traits<_Derived>::BatchesAtCompileTime,
    internal::traits<_Derived>::Flags, const _Derived>
segmentHelper(Index start, Index length, std::false_type) const
{
    //row vector
    CUMAT_ASSERT_ARGUMENT(start >= 0);
    CUMAT_ASSERT_ARGUMENT(start + length <= cols());
    return MatrixBlock<
        typename internal::traits<_Derived>::Scalar,
        1, Dynamic, internal::traits<_Derived>::BatchesAtCompileTime,
        internal::traits<_Derived>::Flags, const _Derived>(
            derived(), 1, length, batches(), 0, start, 0);
}

public:
/**
* \brief Extracts a dynamic-size segment of the vector.
*   Only available for vectors
* \param start the start position of the segment
* \param length the length of the segment
*/
auto
segment(Index start, Index length) const -> decltype(segmentHelper(start, length, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>()))
{
	CUMAT_ERROR_IF_NO_NVCC(segment)
    CUMAT_STATIC_ASSERT(
        (internal::traits<_Derived>::RowsAtCompileTime == 1 || internal::traits<_Derived>::ColsAtCompileTime == 1),
        "segment can only act on compile-time vectors");
    return segmentHelper(start, length, std::integral_constant<bool, internal::traits<_Derived>::ColsAtCompileTime == 1>());
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
* \brief Extracts a dynamic-size segment from the tail of the vector.
* Only available for vectors
* \param length the length of the segment
*/
auto tail(Index length) const -> decltype(segment(0, length))
{
	CUMAT_ERROR_IF_NO_NVCC(tail)
    return segment(std::max(rows(), cols()) - length, length);
}
