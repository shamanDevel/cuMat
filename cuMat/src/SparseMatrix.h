#ifndef __CUMAT_SPARSE_MATRIX__
#define __CUMAT_SPARSE_MATRIX__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "NumTraits.h"
#include "Constants.h"
#include "DevicePointer.h"
#include "MatrixBase.h"
#include "Matrix.h"

CUMAT_NAMESPACE_BEGIN


namespace internal
{
    template <typename _Scalar, int _Batches, int _Flags>
    struct traits<SparseMatrix<_Scalar, _Batches, _Flags> >
    {
        typedef _Scalar Scalar;
        enum
        {
            Flags = _Flags,
            RowsAtCompileTime = Dynamic,
            ColsAtCompileTime = Dynamic,
            BatchesAtCompileTime = _Batches,
            AccessFlags = ReadCwise | WriteCwise | RWCwise
        };
    };
}

/**
 * \brief A sparse matrix in CSC or CSR storage.
 *  The matrix always has a dynamic number of rows and columns, but the batch size can be fixed on compile time.
 *  
 * If the sparse matrix is batches, every batch shares the same sparsity pattern.
 * 
 * \tparam _Scalar the scalar type
 * \tparam _Batches the number of batches on compile time or Dynamic
 * \tparam _Flags the storage mode, must be either \c SparseFlags::CSC or \c SparseFlags::CSR
 */
template<typename _Scalar, int _Batches, int _Flags>
class SparseMatrix : public MatrixBase<SparseMatrix<_Scalar, _Batches, _Flags> >
{
public:

    using Type = SparseMatrix<_Scalar, _Batches, _Flags>;
    typedef MatrixBase<SparseMatrix<_Scalar, _Batches, _Flags> > Base;
    CUMAT_PUBLIC_API
    enum
    {
        TransposedFlags = Flags == SparseFlags::CSR ? RowMajor : ColumnMajor
    };
    using Base::derived;
    using Base::eval_t;

    /**
     * \brief The type of the storage indices.
     * This is fixed to an integer and not using Index, because this is faster for CUDA.
     */
    typedef int StorageIndex;

    typedef Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1, Batches, Flags::ColumnMajor> ScalarVector;

    /**
     * \brief The sparsity pattern to initialize a sparse matrix
     */
    struct SparsityPattern
    {
        Index nnz;
        Index rows;
        Index cols;
        IndexVector IA;
        IndexVector JA;
    };

private:
    /**
     * \brief Number of non-zero elements.
     */
    Index nnz_;
    /**
     * \brief Number of rows in the matrix.
     */
    Index rows_;
    /**
     * \brief Number of columns in the matrix.
     */
    Index cols_;
    /**
     * \brief Number of batches.
     */
    Index batches_;

    /**
     * \brief The (possibly batched) vector with the coefficients of size nnz_ .
     */
    ScalarVector A_;
    /**
    * \brief The inner indices of the coefficients, size nnz_ .
    */
    IndexVector IA_;
    /**
     * \brief The outer indices, size N+1
     */
    IndexVector JA_;

public:

    //----------------------------------
    //  CONSTRUCTORS
    //----------------------------------

    /**
     * \brief Default constructor, SparseMatrix is empty
     */
    SparseMatrix()
        : nnz_(0), rows_(0), cols_(0), batches_(0)
    {}

    /**
     * \brief Initializes this SparseMatrix with the given sparsity pattern and (if not fixed by the template argument)
     * with the given number of batches.
     * The coefficient array is allocated, but uninitialized.
     * 
     * \param sparsityPattern the sparsity pattern
     * \param batches the number of batches
     */
    SparseMatrix(const SparsityPattern& sparsityPattern, Index batches = _Batches)
        : nnz_(sparsityPattern.nnz)
        , rows_(sparsityPattern.rows)
        , cols_(sparsityPattern.cols)
        , batches_(batches)
        , A_(sparsityPattern.nnz, 1, batches) //This also checks if the number of batches is valid
        , IA_(sparsityPattern.IA)
        , JA_(sparsityPattern.JA)
    {
        CUMAT_ASSERT_ARGUMENT(batches > 0);
    }

    /**
     * \brief Returns the sparsity pattern of this matrix.
     * This can be used to create another matrix with the same sparsity pattern.
     * \return The sparsity pattern of this matrix.
     */
    SparsityPattern getSparsityPattern() const
    {
        return { nnz_, rows_, cols_, IA_, JA_ };
    }


    SparseMatrix(const SparseMatrix& other)
        : nnz_(other.nnz_),
          rows_(other.rows_),
          cols_(other.cols_),
          batches_(other.batches_),
          A_(other.A_),
          IA_(other.IA_),
          JA_(other.JA_)
    {
    }

    SparseMatrix(SparseMatrix&& other) noexcept
        : nnz_(other.nnz_),
          rows_(other.rows_),
          cols_(other.cols_),
          batches_(other.batches_),
          A_(std::move(other.A_)),
          IA_(std::move(other.IA_)),
          JA_(std::move(other.JA_))
    {
    }

    SparseMatrix& operator=(const SparseMatrix& other)
    {
        if (this == &other)
            return *this;
        nnz_ = other.nnz_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        batches_ = other.batches_;
        A_ = other.A_;
        IA_ = other.IA_;
        JA_ = other.JA_;
        return *this;
    }

    SparseMatrix& operator=(SparseMatrix&& other) noexcept
    {
        if (this == &other)
            return *this;
        nnz_ = other.nnz_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        batches_ = other.batches_;
        A_ = std::move(other.A_);
        IA_ = std::move(other.IA_);
        JA_ = std::move(other.JA_);
        return *this;
    }


    //----------------------------------
    //  COEFFICIENT ACCESS
    //----------------------------------

    /**
    * \brief Returns the number of rows of this matrix.
    * \return the number of rows
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }

    /**
    * \brief Returns the number of columns of this matrix.
    * \return the number of columns
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_); }

    /**
    * \brief Returns the number of batches of this matrix.
    * \return the number of batches
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }

    /**
    * \brief Returns the number of non-zero coefficients in this matrix
    * \return the number of non-zeros
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index nnz() const { return nnz_; }

    /**
    * \brief Returns the number of non-zero coefficients in this matrix.
    * \return the number of non-zeros
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index size() const { return nnz_; }

    /**
    * \brief Converts from the linear index back to row, column and batch index.
    * Requirement of \c AccessFlags::WriteCwise
    * \param index the linear index
    * \param row the row index (output)
    * \param col the column index (output)
    * \param batch the batch index (output)
    */
    __host__ __device__ CUMAT_STRONG_INLINE void index(Index index, Index& row, Index& col, Index& batch) const
    {
        //TODO
    }

    //----------------------------------
    //  EVALUATION
    //----------------------------------

};

CUMAT_NAMESPACE_END


#endif