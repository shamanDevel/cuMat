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
            AccessFlags = ReadCwise | WriteCwise | (traits<_MatrixType>::AccessFlags & RWCwise ? RWCwise : 0) | (traits<_MatrixType>::AccessFlags & RWCwiseRef ? RWCwiseRef : 0)
		};
        typedef CwiseSrcTag SrcTag;
        typedef DenseDstTag DstTag;
	};

} //end namespace internal

template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, typename _MatrixType>
class MatrixBlock : public CwiseOp<MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType> >
{
public:
	using Type = MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType>;
	using MatrixType = _MatrixType;
	using Base = MatrixBase<MatrixBlock<_Scalar, _Rows, _Columns, _Batches, _Flags, _MatrixType> >;
    CUMAT_PUBLIC_API

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
	__device__ CUMAT_STRONG_INLINE _Scalar& coeff(Index row, Index col, Index batch, Index index)
	{
		return matrix_.coeff(row + start_row_, col + start_column_, batch + start_batch_, -1);
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
	__device__ CUMAT_STRONG_INLINE _Scalar coeff(Index row, Index col, Index batch, Index index) const
	{
		return matrix_.coeff(row + start_row_, col + start_column_, batch + start_batch_, -1);
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
		coeff(i, j, k, -1) = newValue;
	}

	/**
	* \brief Access to the linearized coefficient, read-only.
	* The format of the indexing depends on whether this
	* matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
	* Requirement of \c AccessFlags::RWCwise .
	* \param idx the linearized index of the entry.
	* \return the entry at that index
	*/
	__device__ CUMAT_STRONG_INLINE _Scalar getRawCoeff(Index idx) const
	{
		//This method is quite ineffective at the moment, since it has to convert the values back to row,col,batch
		Index i, j, k;
		index(idx, i, j, k);
		return coeff(i, j, k, -1);
	}

	/**
	* \brief Access to the linearized coefficient, read-only.
	* The format of the indexing depends on whether this
	* matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
	* Requirement of \c AccessFlags::RWCwiseRef .
	* \param index the linearized index of the entry.
	* \return the entry at that index
	*/
	__device__ CUMAT_STRONG_INLINE _Scalar& rawCoeff(Index idx)
	{
		//This method is quite ineffective at the moment, since it has to convert the values back to row,col,batch
		Index i, j, k;
		index(idx, i, j, k);
		return coeff(i, j, k, -1);
	}

	//ASSIGNMENT

	template<typename Derived>
	CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
	{
		CUMAT_ASSERT_ARGUMENT(rows() == expr.rows());
		CUMAT_ASSERT_ARGUMENT(cols() == expr.cols());
		CUMAT_ASSERT_ARGUMENT(batches() == expr.batches());
		//expr.template evalTo<Type, AssignmentMode::ASSIGN>(*this);
        internal::Assignment<Type, Derived, AssignmentMode::ASSIGN, internal::DenseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());
		return *this;
	}

#define CUMAT_COMPOUND_ASSIGNMENT(op, mode)                                                             \
    /**                                                                                                 \
    * \brief Compound-assignment with evaluation, modifies this matrix in-place.                        \
    * Warning: if this matrix shares the data with another matrix, this matrix is modified as well.     \
    * If you don't intend this, call \ref makeExclusiveUse() first.                                     \
    *                                                                                                   \
    * No broadcasting is supported, use the verbose \code mat = mat + expr \endcode instead.            \
    * Further, not all expressions might support inplace-assignment.                                    \
    *                                                                                                   \
    * \tparam Derived the type of the other expression                                                  \
    * \param expr the other expression                                                                  \
    * \return                                                                                           \
    */                                                                                                  \
    template<typename Derived>                                                                          \
    CUMAT_STRONG_INLINE Type& op (const MatrixBase<Derived>& expr)                                      \
    {                                                                                                   \
        CUMAT_ASSERT_ARGUMENT(rows() == expr.rows());													\
		CUMAT_ASSERT_ARGUMENT(cols() == expr.cols());													\
		CUMAT_ASSERT_ARGUMENT(batches() == expr.batches());												\
		internal::Assignment<Type, Derived, AssignmentMode:: mode, internal::DenseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());	\
		return *this;                                                                                 \
    }

    CUMAT_COMPOUND_ASSIGNMENT(operator+=, ADD)
    CUMAT_COMPOUND_ASSIGNMENT(operator-=, SUB)
    //CUMAT_COMPOUND_ASSIGNMENT(operator*=, MUL) //multiplication is ambigious: do you want cwise or matrix multiplication?
    CUMAT_COMPOUND_ASSIGNMENT(operator/=, DIV)
    CUMAT_COMPOUND_ASSIGNMENT(operator%=, MOD)
    CUMAT_COMPOUND_ASSIGNMENT(operator&=, AND)
    CUMAT_COMPOUND_ASSIGNMENT(operator|=, OR)

	/**
	* \brief Explicit overloading of \c operator*= for scalar right hand sides.
	* This is needed to disambiguate the difference between component-wise operations and matrix operations.
	* All other compount-assignment operators (+=, -=, /=, ...) act component-wise.
	*
	* \tparam S the type of the scalar
	* \param scalar the scalar value
	* \return *this
	*/
	template<
		typename S,
		typename T = typename std::enable_if<CUMAT_NAMESPACE internal::canBroadcast<Scalar, S>::value, Type>::type >
	CUMAT_STRONG_INLINE T& operator*= (const S& scalar)
	{
		//Type::Constant(rows(), cols(), batches(), scalar).template evalTo<Type, AssignmentMode::MUL>(*this);
		using Expr = decltype(Type::Constant(rows(), cols(), batches(), scalar));
		internal::Assignment<Type, Expr, AssignmentMode::MUL, internal::DenseDstTag, typename internal::traits<Expr>::SrcTag>::assign(*this, Type::Constant(rows(), cols(), batches(), scalar));
		return *this;
	}

#undef CUMAT_COMPOUND_ASSIGNMENT
};


CUMAT_NAMESPACE_END

#endif
