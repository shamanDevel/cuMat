#ifndef __CUMAT_SPARSE_MATRIX_BASE__
#define __CUMAT_SPARSE_MATRIX_BASE__

#include "Macros.h"
#include "ForwardDeclarations.h"

CUMAT_NAMESPACE_BEGIN

/**
* \brief The sparsity pattern to initialize a sparse matrix
*/
struct SparsityPattern
{
    /**
    * \brief The type of the storage indices.
    * This is fixed to an integer and not using Index, because this is faster for CUDA.
    */
    typedef int StorageIndex;

    typedef Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> IndexVector;
    typedef Matrix<const StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> ConstIndexVector;

    Index nnz;
    Index rows;
    Index cols;
    /** \brief Inner indices, size=nnz */
    IndexVector IA;
    /** \brief Outer indices, size=N+1 */
    IndexVector JA;

    /**
    * \brief Checks with assertions that this SparsityPattern is valid.
    * \tparam _SparseFlags the sparse format, can be either CSR or CSC.
    */
    template<int _SparseFlags>
    void assertValid() const
    {
        CUMAT_STATIC_ASSERT(_SparseFlags == SparseFlags::CSR || _SparseFlags == SparseFlags::CSC,
            "SparseFlags must be either CSR or CSC");
        if (_SparseFlags == SparseFlags::CSC) { CUMAT_ASSERT_DIMENSION(JA.size() == cols + 1) }
        else /*CSR*/ { CUMAT_ASSERT_DIMENSION(JA.size() == rows + 1); }
        CUMAT_ASSERT_DIMENSION(rows > 0);
        CUMAT_ASSERT_DIMENSION(cols > 0);
        CUMAT_ASSERT_DIMENSION(cols > 0);
        CUMAT_ASSERT_DIMENSION(IA.size() == nnz);
    }
};

template<typename _Derived>
class SparseMatrixBase : public MatrixBase<_Derived>
{
public:
    typedef _Derived Type;
    typedef MatrixBase<_Derived> Base;
    CUMAT_PUBLIC_API
    enum
    {
        SparseFlags = internal::traits<_Derived>::SparseFlags
    };

    /**
    * \brief The type of the storage indices.
    * This is fixed to an integer and not using Index, because this is faster for CUDA.
    */
    typedef SparsityPattern::StorageIndex StorageIndex;

    typedef SparsityPattern::IndexVector IndexVector;
    typedef SparsityPattern::ConstIndexVector ConstIndexVector;

protected:
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
    * \brief The inner indices of the coefficients, size nnz_ .
    */
    IndexVector IA_;
    /**
    * \brief The outer indices, size N+1
    */
    IndexVector JA_;

public:
    SparseMatrixBase()
        : nnz_(0), rows_(0), cols_(0), batches_(0)
    {}

    SparseMatrixBase(const SparsityPattern& sparsityPattern, Index batches)
        : nnz_(sparsityPattern.nnz)
        , rows_(sparsityPattern.rows)
        , cols_(sparsityPattern.cols)
        , batches_(batches)
        , IA_(sparsityPattern.IA)
        , JA_(sparsityPattern.JA)
    {
        sparsityPattern.assertValid<SparseFlags>();
        CUMAT_ASSERT(CUMAT_IMPLIES(Batches == Dynamic, Batches == batches) &&
            "compile-time batch count specified, but does not match runtime batch count");
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

    SparseMatrixBase(const SparseMatrixBase& other)
        : nnz_(other.nnz_),
        rows_(other.rows_),
        cols_(other.cols_),
        batches_(other.batches_),
        IA_(other.IA_),
        JA_(other.JA_)
    {
    }

    SparseMatrixBase(SparseMatrixBase&& other) noexcept
        : nnz_(other.nnz_),
        rows_(other.rows_),
        cols_(other.cols_),
        batches_(other.batches_),
        IA_(std::move(other.IA_)),
        JA_(std::move(other.JA_))
    {}

    SparseMatrixBase& operator=(const SparseMatrixBase& other)
    {
        if (this == &other)
            return *this;
        nnz_ = other.nnz_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        batches_ = other.batches_;
        IA_ = other.IA_;
        JA_ = other.JA_;
        return *this;
    }

    SparseMatrixBase& operator=(SparseMatrixBase&& other) noexcept
    {
        if (this == &other)
            return *this;
        nnz_ = other.nnz_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        batches_ = other.batches_;
        IA_ = std::move(other.IA_);
        JA_ = std::move(other.JA_);
        return *this;
    }

    bool isInitialized() const
    {
        return rows_ > 0 && cols_ > 0 && batches_ > 0;
    }

    __host__ __device__ CUMAT_STRONG_INLINE const IndexVector& getInnerIndices() { return IA_; }
    __host__ __device__ CUMAT_STRONG_INLINE ConstIndexVector getInnerIndices() const { return ConstIndexVector(IA_.dataPointer(), IA_.rows(), IA_.cols(), IA_.batches()); }
    __host__ __device__ CUMAT_STRONG_INLINE const IndexVector& getOuterIndices() { return JA_; }
    __host__ __device__ CUMAT_STRONG_INLINE ConstIndexVector getOuterIndices() const { return ConstIndexVector(JA_.dataPointer(), JA_.rows(), JA_.cols(), JA_.batches()); }

    /**
    * \brief Returns the number of rows of this matrix.
    * \return the number of rows
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }

    /**
    * \brief Returns the number of columns of this matrix.
    * \return the number of columns
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_; }

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

    __host__ __device__ CUMAT_STRONG_INLINE Index outerSize() const
    {
        return (SparseFlags == SparseFlags::CSC) ? cols() : rows();
    }
};

CUMAT_NAMESPACE_END

#endif