#ifndef __CUMAT_SPARSE_MATRIX_BASE__
#define __CUMAT_SPARSE_MATRIX_BASE__

#include "Macros.h"
#include "ForwardDeclarations.h"

CUMAT_NAMESPACE_BEGIN

/**
* \brief The sparsity pattern to initialize a sparse matrix.
* 
* The exact data depends on the sparse storage type.
* But the pattern must conform the following requirements:
* <code>
*	typedef ... StorageIndex;
*	Index rows;
*	Index cols;
*	Index getNNZ() const;
*	void assertValid() const;
*	template<typename _Scalar, int _Batches> using DataMatrix = ...;
*	SparsityPattern deepClone() const;
* </code>
* 
* \tparam _SparseFlags the sparse storage type of type \ref SparseFlags.
*/
template<int _SparseFlags>
struct SparsityPattern;

template<>
struct SparsityPattern<SparseFlags::CSC>
{
    /**
    * \brief The type of the storage indices.
    * This is fixed to an integer and not using Index, because this is faster for CUDA.
    */
    typedef int StorageIndex;

    typedef Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> IndexVector;
    typedef const Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> ConstIndexVector; //TODO: this is no proper const-correctness
	template<typename _Scalar, int _Batches> using DataMatrix = Matrix<_Scalar, Dynamic, 1, _Batches, Flags::ColumnMajor>;

    Index nnz;
    Index rows;
    Index cols;
    /** \brief Inner indices, size=nnz */
    IndexVector IA;
    /** \brief Outer indices, size=N+1 */
    IndexVector JA;

    /**
    * \brief Checks with assertions that this SparsityPattern is valid.
    */
    void assertValid() const
    {
		CUMAT_ASSERT_DIMENSION(JA.size() == cols + 1);
        CUMAT_ASSERT_DIMENSION(rows > 0);
        CUMAT_ASSERT_DIMENSION(cols > 0);
        CUMAT_ASSERT_DIMENSION(cols > 0);
        CUMAT_ASSERT_DIMENSION(IA.size() == nnz);
    }
	/**
	 * \return the number of non-zero entries
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index getNNZ() const { return nnz; }
	/**
	 * \brief Allocates the data matrix for this sparsity type with the specified number of batches.
	 */
	template<typename _Scalar, int _Batches>
	DataMatrix<_Scalar, _Batches> allocateDataMatrix(Index batches) const
    {
		return DataMatrix<_Scalar, _Batches>(nnz, 1, batches);
    }

	/**
	 * \return a deep clone of this sparsity pattern
	 */
	SparsityPattern<SparseFlags::CSC> deepClone() const
    {
		SparsityPattern<SparseFlags::CSC> clone;
		clone.nnz = nnz;
		clone.rows = rows;
		clone.cols = cols;
		clone.IA = IA.deepClone();
		clone.JA = JA.deepClone();
		return clone;
    }
};

template<>
struct SparsityPattern<SparseFlags::CSR>
{
	/**
	* \brief The type of the storage indices.
	* This is fixed to an integer and not using Index, because this is faster for CUDA.
	*/
	typedef int StorageIndex;

	typedef Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> IndexVector;
	typedef const Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> ConstIndexVector; //TODO: this is no proper const-correctness
	template<typename _Scalar, int _Batches> using DataMatrix = Matrix<_Scalar, Dynamic, 1, _Batches, Flags::ColumnMajor>;

	Index nnz;
	Index rows;
	Index cols;
	/** \brief Inner indices, size=nnz */
	IndexVector IA;
	/** \brief Outer indices, size=N+1 */
	IndexVector JA;

	/**
	* \brief Checks with assertions that this SparsityPattern is valid.
	*/
	void assertValid() const
	{
		CUMAT_ASSERT_DIMENSION(JA.size() == rows + 1);
		CUMAT_ASSERT_DIMENSION(rows > 0);
		CUMAT_ASSERT_DIMENSION(cols > 0);
		CUMAT_ASSERT_DIMENSION(cols > 0);
		CUMAT_ASSERT_DIMENSION(IA.size() == nnz);
	}
	/**
	 * \return the number of non-zero entries
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index getNNZ() const { return nnz; }
	/**
	 * \brief Allocates the data matrix for this sparsity type with the specified number of batches.
	 */
	template<typename _Scalar, int _Batches>
	DataMatrix<_Scalar, _Batches> allocateDataMatrix(Index batches) const
	{
		return DataMatrix<_Scalar, _Batches>(nnz, 1, batches);
	}

	/**
	 * \return a deep clone of this sparsity pattern
	 */
	SparsityPattern<SparseFlags::CSR> deepClone() const
	{
		SparsityPattern<SparseFlags::CSR> clone;
		clone.nnz = nnz;
		clone.rows = rows;
		clone.cols = cols;
		clone.IA = IA.deepClone();
		clone.JA = JA.deepClone();
		return clone;
	}
};

template<>
struct SparsityPattern<SparseFlags::ELLPACK>
{
	/**
	* \brief The type of the storage indices.
	* This is fixed to an integer and not using Index, because this is faster for CUDA.
	*/
	typedef int StorageIndex;

	typedef Matrix<StorageIndex, Dynamic, Dynamic, 1, Flags::ColumnMajor> IndexMatrix;
	typedef const Matrix<StorageIndex, Dynamic, Dynamic, 1, Flags::ColumnMajor> ConstIndexMatrix;
	template<typename _Scalar, int _Batches> using DataMatrix = Matrix<_Scalar, Dynamic, Dynamic, _Batches, Flags::ColumnMajor>;

	Index rows;
	Index cols;
	Index nnzPerRow;
	/** \brief Index matrix, -1 indicates no data. */
	IndexMatrix indices;

	/**
	* \brief Checks with assertions that this SparsityPattern is valid.
	*/
	void assertValid() const
	{
		CUMAT_ASSERT_DIMENSION(indices.rows() == rows);
		CUMAT_ASSERT_DIMENSION(indices.cols() == nnzPerRow);
		CUMAT_ASSERT_DIMENSION(rows > 0);
		CUMAT_ASSERT_DIMENSION(cols > 0);
		CUMAT_ASSERT_DIMENSION(cols > 0);
		CUMAT_ASSERT_DIMENSION(nnzPerRow > 0);
		CUMAT_ASSERT_DIMENSION(nnzPerRow <= cols);
	}
	/**
	 * \return the number of non-zero entries
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index getNNZ() const { return rows*nnzPerRow; }
	/**
	 * \brief Allocates the data matrix for this sparsity type with the specified number of batches.
	 */
	template<typename _Scalar, int _Batches>
	DataMatrix<_Scalar, _Batches> allocateDataMatrix(Index batches) const
	{
		return DataMatrix<_Scalar, _Batches>(rows, nnzPerRow, batches);
	}

	/**
	 * \return a deep clone of this sparsity pattern
	 */
	SparsityPattern<SparseFlags::ELLPACK> deepClone() const
	{
		SparsityPattern<SparseFlags::ELLPACK> clone;
		clone.nnzPerRow = nnzPerRow;
		clone.rows = rows;
		clone.cols = cols;
		clone.indices = indices.deepClone();
		return clone;
	}
};


namespace internal
{
	/**
	 * index computations for sparse matrices
	 */
	template <int _SparseFlags>
	struct SparseMatrixIndexEvaluator
	{
		using Sparsity = SparsityPattern<_SparseFlags>;
		/**
		 * \brief Converts linear index of nnz to row+col+batch index. This is required for AccessFlags::WriteCwise
		 */
		static __device__ void linearToCoords(const Sparsity& sparsity, Index index, Index& row, Index& col, Index& batch);
		/**
		 * \brief Converts coordinate indices to the linear index. This is required for AccessFlags::ReadCwise
		 */
		static __device__ Index coordsToLinear(const Sparsity& sparsity, Index row, Index col, Index batch);
	};

	//CSR specialization
	template <>
	struct SparseMatrixIndexEvaluator<SparseFlags::CSR>
	{
		using Sparsity = SparsityPattern<SparseFlags::CSR>;
		/**
		 * \brief Converts linear index of nnz to row+col+batch index. This is required for AccessFlags::WriteCwise
		 */
		static __device__ void linearToCoords(const Sparsity& sparsity, Index index, Index& row, Index& col, Index& batch)
		{
			batch = index / sparsity.nnz;
			index = index % sparsity.nnz;
			//find row
			int start = sparsity.JA.getRawCoeff(0);
			for (int i = 0; i < sparsity.rows; ++i)
			{
				int end = sparsity.JA.getRawCoeff(i + 1);
				if (index < end)
				{
					//index is inside this row, search for column
					row = i;
					col = sparsity.IA.getRawCoeff(index);
					return;
				}
				start = end;
			}
		}
		/**
		 * \brief Converts coordinate indices to the linear index. This is required for AccessFlags::ReadCwise
		 */
		static __device__ Index coordsToLinear(const Sparsity& sparsity, Index row, Index col, Index batch)
		{
			const int start = sparsity.JA.getRawCoeff(row);
			const int end = sparsity.JA.getRawCoeff(row + 1);
			for (int i = start; i < end; ++i)
			{
				int c = sparsity.IA.getRawCoeff(i);
				if (c == col) {
					const int idx = i + sparsity.nnz * batch;
					return idx;
				}
			}
			return -1;
		}
	};

	//CSC specialization
	template <>
	struct SparseMatrixIndexEvaluator<SparseFlags::CSC>
	{
		using Sparsity = SparsityPattern<SparseFlags::CSC>;
		/**
		 * \brief Converts linear index of nnz to row+col+batch index. This is required for AccessFlags::WriteCwise
		 */
		static __device__ void linearToCoords(const Sparsity& sparsity, Index index, Index& row, Index& col, Index& batch)
		{
			batch = index / sparsity.nnz;
			index = index % sparsity.nnz;
			//find column
			int start = sparsity.JA.getRawCoeff(0);
			for (int i = 0; i < sparsity.cols; ++i)
			{
				int end = sparsity.JA.getRawCoeff(i + 1);
				if (index < end)
				{
					//index is inside this row, search for column
					col = i;
					row = sparsity.IA.getRawCoeff(index);
					return;
				}
				start = end;
			}
		}
		/**
		 * \brief Converts coordinate indices to the linear index. This is required for AccessFlags::ReadCwise
		 */
		static __device__ Index coordsToLinear(const Sparsity& sparsity, Index row, Index col, Index batch)
		{
			const int start = sparsity.JA.getRawCoeff(col);
			const int end = sparsity.JA.getRawCoeff(col + 1);
			for (int i = start; i < end; ++i)
			{
				int r = sparsity.IA.getRawCoeff(i);
				if (r == row) return i + sparsity.nnz * batch;
			}
			return -1;
		}
	};

	//ELLPACK specialization
	template <>
	struct SparseMatrixIndexEvaluator<SparseFlags::ELLPACK>
	{
		using Sparsity = SparsityPattern<SparseFlags::ELLPACK>;
		/**
		 * \brief Converts linear index of nnz to row+col+batch index. This is required for AccessFlags::WriteCwise
		 */
		static __device__ void linearToCoords(const Sparsity& sparsity, Index index, Index& row, Index& col, Index& batch)
		{
			Index nnz = sparsity.rows * sparsity.nnzPerRow;
			batch = index / nnz;
			index = index % nnz;
			col = index / sparsity.nnzPerRow;
			row = index % sparsity.nnzPerRow;
			col = sparsity.indices.coeff(row, col, 0, -1);
		}
		/**
		 * \brief Converts coordinate indices to the linear index. This is required for AccessFlags::ReadCwise
		 */
		static __device__ Index coordsToLinear(const Sparsity& sparsity, Index row, Index col, Index batch)
		{
			for (int i = 0; i < sparsity.nnzPerRow; ++i)
			{
				int c = sparsity.indices.coeff(row, i, 0, -1);
				if (c == col) {
					const int idx = row + sparsity.rows * (i + sparsity.nnzPerRow * batch);
					return idx;
				}
			}
			return -1;
		}
	};
}


/**
 * \brief Base class of all sparse matrices.
 *  It stores the sparsity pattern.
 * \tparam _Derived the type of the derived class
 * \sa SparseMatrix
 */
template<typename _Derived>
class SparseMatrixBase : public MatrixBase<_Derived>
{
public:
    typedef _Derived Type;
    typedef MatrixBase<_Derived> Base;
    CUMAT_PUBLIC_API
    enum
    {
		/**
		 * \brief Sparsity pattern
		 */
        SFlags = internal::traits<_Derived>::SFlags
    };

    /**
    * \brief The type of the storage indices.
    * This is fixed to an integer and not using Index, because this is faster for CUDA.
    */
    using StorageIndex = typename SparsityPattern<SFlags>::StorageIndex;

protected:
	SparsityPattern<SFlags> sparsity_;
	/**
	* \brief Number of batches.
	*/
	Index batches_;

public:
    SparseMatrixBase()
		: batches_(0)
    {}

    SparseMatrixBase(const SparsityPattern<SFlags>& sparsityPattern, Index batches)
        : sparsity_(sparsityPattern)
        , batches_(batches)
    {
        sparsityPattern.assertValid();
        CUMAT_ASSERT(CUMAT_IMPLIES(Batches == Dynamic, Batches == batches) &&
            "compile-time batch count specified, but does not match runtime batch count");
    }

    /**
    * \brief Returns the sparsity pattern of this matrix.
    * This can be used to create another matrix with the same sparsity pattern.
    * \return The sparsity pattern of this matrix.
    */
	__host__ __device__ CUMAT_STRONG_INLINE const SparsityPattern<SFlags>& getSparsityPattern() const
    {
		return sparsity_;
    }

    /**
    * \brief Checks if this matrix is initialized with a sparsity pattern and can therefore be used in expressions.
    */
    bool isInitialized() const
    {
        return sparsity_.rows > 0 && sparsity_.cols > 0 && batches_ > 0;
    }

    /**
    * \brief Returns the number of rows of this matrix.
    * \return the number of rows
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return sparsity_.rows; }

    /**
    * \brief Returns the number of columns of this matrix.
    * \return the number of columns
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return sparsity_.cols; }

    /**
    * \brief Returns the number of batches of this matrix.
    * \return the number of batches
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }

    /**
    * \brief Returns the number of non-zero coefficients in this matrix
    * \return the number of non-zeros
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index nnz() const { return sparsity_.getNNZ() * batches_; }

    /**
    * \brief Returns the number of non-zero coefficients in this matrix.
    * \return the number of non-zeros
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index size() const { return sparsity_.getNNZ() * batches_; }

    /**
    * \brief Returns the outer size of the matrix.
    * This is the number of rows for CSR, or the number of columns for CSC.
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index outerSize() const
    {
        return (SFlags == SparseFlags::CSC) ? cols() : rows();
    }

    /**
     * \brief Sparse coefficient access.
     * For a sparse evaluation of this matrix (and subclasses), first the inner and outer indices 
     * (getInnerIndices() and getOuterIndices()) are queried and then looped over the inner indices.
     * To query the value at a specific sparse index, this method is called.
     * Row, column and batch are the current position and index is the linear index in the data array
     * (same index as where the inner index was queried plus the batch offset).
     * 
     * For a sparse matrix stored in memory (class SparseMatrix), this method simply
     * delegates to <tt>getData().getRawCoeff(index)</tt>, thus directly accessing
     * the linear data without any overhead.
     * 
     * For sparse expressions (class SparseMatrixExpression), the functor is evaluated
     * at the current position row,col,batch. The linear index is passed on unchanged,
     * thus allowing subsequent sparse matrix an efficient access with \ref SparseMatrix::linear().
     * 
     * Each implementation of SparseMatrixBase must implement this routine.
     * Currently, the only real usage of this method is the sparse matrix-vector product.
     * 
     * \param row 
     * \param col 
     * \param batch 
     * \param index 
     * \return 
     */
    __device__ CUMAT_STRONG_INLINE const Scalar& getSparseCoeff(Index row, Index col, Index batch, Index index) const
    {
        return derived().getSparseCoeff(row, col, batch, index);
    }
};

CUMAT_NAMESPACE_END

#endif