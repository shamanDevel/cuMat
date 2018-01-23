#ifndef __CUMAT_MATRIX_BLOCK_H__
#define __CUMAT_MATRIX_BLOCK_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "MatrixBase.h"
#include "CwiseOp.h"

CUMAT_NAMESPACE_BEGIN

namespace internal {
	template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _MatrixType>
	struct traits<MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType> >
	{
		typedef _Scalar Scalar;
		enum
		{
			Flags = _Flags,
			RowsAtCompileTime = _Rows,
			ColsAtCompileTime = _Columns,
			BatchesAtCompileTime = _Batches,
            AccessFlags = ReadCwise | WriteCwise
		};
	};

} //end namespace internal

template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _MatrixType>
class MatrixBlock : public CwiseOp<MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType> >
{
public:
	enum
	{
		Flags = _Flags,
		Rows = _Rows,
		Columns = _Columns,
		Batches = _Batches
	};
	using Scalar = _Scalar;
	using Type = MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType>;

	using MatrixType = _MatrixType;
	using Base = MatrixBase<MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType> >;
	using Base::eval_t;

protected:

	MatrixType matrix_;
	const Index rows_;
	const Index columns_;
	const Index batches_;
	const Index start_row_;
	const Index start_column_;
	const Index start_batch_;

public:
	MatrixBlock(MatrixType& matrix, Index rows, Index columns, Index batches, Index start_row, Index start_column, Index start_batch)
		: matrix_(matrix)
		, rows_(rows)
		, columns_(columns)
		, batches_(batches)
		, start_row_(start_row)
		, start_column_(start_column)
		, start_batch_(start_batch)
	{}

	/**
	* \brief Returns the number of rows of this matrix.
	* \return the number of rows
	*/
	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }

	/**
	* \brief Returns the number of columns of this matrix.
	* \return the number of columns
	*/
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return columns_; }

	/**
	* \brief Returns the number of batches of this matrix.
	* \return the number of batches
	*/
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }

	/**
	* \brief Converts from the linear index back to row, column and batch index
	* \param index the linear index
	* \param row the row index (output)
	* \param col the column index (output)
	* \param batch the batch index (output)
	*/
	__host__ __device__ CUMAT_STRONG_INLINE void index(Index index, Index& row, Index& col, Index& batch) const
	{
		if (CUMAT_IS_ROW_MAJOR(Flags)) {
			batch = index / (rows() * cols());
			index -= batch * rows() * cols();
			row = index / cols();
			index -= row * cols();
			col = index;
		}
		else {
			batch = index / (rows() * cols());
			index -= batch * rows() * cols();
			col = index / rows();
			index -= col * rows();
			row = index;
		}
	}

	/**
	* \brief Accesses the coefficient at the specified coordinate for reading and writing.
	* If the device supports it (CUMAT_ASSERT_CUDA is defined), the
	* access is checked for out-of-bound tests by assertions.
	* \param row the row index
	* \param col the column index
	* \param batch the batch index
	* \return a reference to the entry
	*/
	__device__ CUMAT_STRONG_INLINE _Scalar& coeff(Index row, Index col, Index batch)
	{
		return matrix_.coeff(row + start_row_, col + start_column_, batch + start_batch_);
	}
	/**
	* \brief Accesses the coefficient at the specified coordinate for reading.
	* If the device supports it (CUMAT_ASSERT_CUDA is defined), the
	* access is checked for out-of-bound tests by assertions.
	* \param row the row index
	* \param col the column index
	* \param batch the batch index
	* \return a read-only reference to the entry
	*/
	__device__ CUMAT_STRONG_INLINE const _Scalar& coeff(Index row, Index col, Index batch) const
	{
		return matrix_.coeff(row + start_row_, col + start_column_, batch + start_batch_);
	}

	/**
	* \brief Access to the linearized coefficient.
	* The format of the indexing depends on whether this
	* matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
	* \param idx the linearized index of the entry.
	* \param newValue the new value at that index
	*/
	__device__ CUMAT_STRONG_INLINE void setRawCoeff(Index idx, const _Scalar& newValue)
	{
		//This method is quite ineffective at the moment, since it has to convert the values back to row,col,batch
		Index i, j, k;
		index(idx, i, j, k);
		coeff(i, j, k) = newValue;
	}

	//ASSIGNMENT

	template<typename Derived>
	CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
	{
		CUMAT_ASSERT_ARGUMENT(rows() == expr.rows());
		CUMAT_ASSERT_ARGUMENT(cols() == expr.cols());
		CUMAT_ASSERT_ARGUMENT(batches() == expr.batches());
		expr.evalTo(*this);
		return *this;
	}
};


CUMAT_NAMESPACE_END

#endif
