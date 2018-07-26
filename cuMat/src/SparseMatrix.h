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
    template <typename _Scalar, int _Batches, int _SparseFlags>
    struct traits<SparseMatrix<_Scalar, _Batches, _SparseFlags> >
    {
        typedef _Scalar Scalar;
        enum
        {
            Flags = ColumnMajor, //always use ColumnMajor when evaluated to dense stuff
            SparseFlags = _SparseFlags,
            RowsAtCompileTime = Dynamic,
            ColsAtCompileTime = Dynamic,
            BatchesAtCompileTime = _Batches,
            AccessFlags = ReadCwise | WriteCwise | RWCwise | RWCwiseRef
        };
        typedef CwiseSrcTag SrcTag;
        typedef SparseDstTag DstTag;
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
 * \tparam _SparseFlags the storage mode, must be either \c SparseFlags::CSC or \c SparseFlags::CSR
 */
template<typename _Scalar, int _Batches, int _SparseFlags>
class SparseMatrix : public MatrixBase<SparseMatrix<_Scalar, _Batches, _SparseFlags> >
{
    CUMAT_STATIC_ASSERT(_SparseFlags == SparseFlags::CSR || _SparseFlags == SparseFlags::CSC,
        "SparseFlags must be either CSR or CSC");
public:

    using Type = SparseMatrix<_Scalar, _Batches, _SparseFlags>;
    typedef MatrixBase<SparseMatrix<_Scalar, _Batches, _SparseFlags> > Base;
    CUMAT_PUBLIC_API_NO_METHODS
    using Base::derived;
    enum
    {
        SparseFlags = _SparseFlags
    };

    /**
     * \brief The type of the storage indices.
     * This is fixed to an integer and not using Index, because this is faster for CUDA.
     */
    typedef int StorageIndex;

    typedef Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> IndexVector;
    typedef Matrix<Scalar, Dynamic, 1, Batches, Flags::ColumnMajor> ScalarVector;
    typedef const Matrix<StorageIndex, Dynamic, 1, 1, Flags::ColumnMajor> ConstIndexVector;
    typedef const Matrix<Scalar, Dynamic, 1, Batches, Flags::ColumnMajor> ConstScalarVector;

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

        /**
         * \brief Checks with assertions that this SparsityPattern is valid.
         */
        void assertValid() const
        {
            if (_SparseFlags == SparseFlags::CSC) { CUMAT_ASSERT_DIMENSION(JA.size() == cols+1) }
            else /*CSR*/ { CUMAT_ASSERT_DIMENSION(JA.size() == rows+1); }
            CUMAT_ASSERT_DIMENSION(rows > 0);
            CUMAT_ASSERT_DIMENSION(cols > 0);
            CUMAT_ASSERT_DIMENSION(cols > 0);
            CUMAT_ASSERT_DIMENSION(IA.size() == nnz);
        }
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

    ~SparseMatrix() = default;

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
        sparsityPattern.assertValid();
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

    bool isInitialized() const
    {
        return rows_ > 0 && cols_ > 0 && batches_ > 0;
    }

    __host__ __device__ CUMAT_STRONG_INLINE ScalarVector getData() { return A_; }
    __host__ __device__ CUMAT_STRONG_INLINE ConstScalarVector getData() const { return A_; }
    __host__ __device__ CUMAT_STRONG_INLINE IndexVector getInnerIndices() { return IA_; }
    __host__ __device__ CUMAT_STRONG_INLINE ConstIndexVector getInnerIndices() const { return IA_; }
    __host__ __device__ CUMAT_STRONG_INLINE IndexVector getOuterIndices() { return JA_; }
    __host__ __device__ CUMAT_STRONG_INLINE ConstIndexVector getOuterIndices() const { return JA_; }

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
        return (_SparseFlags == SparseFlags::CSC) ? cols() : rows();
    }

    /**
     * \brief Accesses a single entry, performs a search for the specific entry.
     * This is required by the \code AccessFlag::CwiseRead \endcode,
     * needed so that a sparse matrix can be used 
     * \param row 
     * \param col 
     * \param batch 
     * \return 
     */
    __device__ Scalar coeff(Index row, Index col, Index batch, Index /*linear*/) const
    {
        //TODO: optimize with a binary search and early bound checks
        if (_SparseFlags == SparseFlags::CSC)
        {
            int start = JA_.getRawCoeff(col);
            int end = JA_.getRawCoeff(col + 1);
            for (int i=start; i<end; ++i)
            {
                int r = IA_.getRawCoeff(i);
                if (r == row) return A_.coeff(i, 0, batch, -1);
            }
        } else //CSR
        {
            int start = JA_.getRawCoeff(row);
            int end = JA_.getRawCoeff(row + 1);
            for (int i = start; i<end; ++i)
            {
                int c = IA_.getRawCoeff(i);
                if (c == col) return A_.coeff(i, 0, batch, -1);
            }
        }
        return Scalar(0);
    }
    //TODO: optimized coefficient access with the use of 'linear'

    /**
    * \brief Access to the linearized coefficient, write-only.
    * The format of the indexing depends on whether this
    * matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
    * Requirement of \c AccessFlags::WriteCwise
    * \param index the linearized index of the entry.
    * \param newValue the new value at that entry
    */
    __device__ CUMAT_STRONG_INLINE void setRawCoeff(Index index, const _Scalar& newValue)
    {
        A_.setRawCoeff(index, newValue);
    }

    /**
    * \brief Access to the linearized coefficient, read-only.
    * The format of the indexing depends on whether this
    * matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
    * Requirement of \c AccessFlags::RWCwise .
    * \param index the linearized index of the entry.
    * \return the entry at that index
    */
    __device__ CUMAT_STRONG_INLINE const _Scalar& getRawCoeff(Index index) const
    {
        return A_.getRawCoeff(index);
    }

    /**
    * \brief Access to the linearized coefficient, read-only.
    * The format of the indexing depends on whether this
    * matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
    * Requirement of \c AccessFlags::RWCwiseRef .
    * \param index the linearized index of the entry.
    * \return the entry at that index
    */
    __device__ CUMAT_STRONG_INLINE _Scalar& rawCoeff(Index index)
    {
        return A_.rawCoeff(index);
    }

    void direct() const
    {
        throw std::exception("Not supported yet");
    }

    //----------------------------------
    //  EVALUATION
    //----------------------------------

    typedef Type eval_t;

    eval_t eval() const
    {
        return eval_t(derived()); //A No-Op
    }

    /**
    * \brief Checks if the this matrix has exclusive use to the underlying data, i.e. no other matrix expression shares the data.
    * \b Note: The data is tested, the sparsity pattern might still be shared!
    * This allows to check if this matrix is used in other expressions because that increments the internal reference counter.
    * If the internal reference counter is one, the matrix is nowhere else copied and this method returns true.
    *
    * This is used to determine if the matrix can be modified inplace.
    * \return
    */
    CUMAT_STRONG_INLINE bool isExclusiveUse() const
    {
        return A_.dataPointer().getCounter() == 1;
    }

    /**
    * \brief Checks if the underlying data is used by an other matrix expression and if so,
    * copies the data so that this matrix is the exclusive user of that data.
    * \b Note: The data is tested, the sparsity pattern might still be shared!
    *
    * This method has no effect if \ref isExclusiveUse is already true.
    *
    * Postcondition: <code>isExclusiveUse() == true</code>
    */
    void makeExclusiveUse()
    {
        if (isExclusiveUse()) return;
        A_ = A_.deepClone();
        assert(isExclusiveUse());
    }

    /**
    * \brief Evaluation assignment, new memory is allocated for the result data.
    * Exception: if it can be guarantered, that the memory is used exclusivly (\ref isExclusiveUse() returns true).
    * 
    * \b Important: The sparsity pattern is kept and determins the non-zero entries where the expression
    * is evaluated.
    * 
    * Further, this assignment throws an std::runtime_error if the sparsity pattern 
    * was not initialized (SparseMatrix created by the default constructor),
    * or if the assignment would require resizing.
    * 
    * \tparam Derived
    * \param expr the expression to evaluate into this matrix.
    * \return *this
    */
    template<typename Derived>
    CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
    {
        if (rows() != expr.rows() || cols() != expr.cols() || batches() != expr.batches())
        {
            throw std::runtime_error("The matrix size of the expression does not match this size, dynamic resizing of a SparseMatrix is not supported");
        }
        if (!isInitialized())
        {
            throw std::runtime_error("The sparsity pattern of this SparseMatrix has not been initialized, can't assign to this matrix");
        }
        makeExclusiveUse();
        internal::Assignment<Type, Derived, AssignmentMode::ASSIGN, internal::SparseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());
        return *this;
    }
    //No evaluation constructor

    /**
    * \brief Forces inplace assignment.
    * The only usecase is <code>matrix.inplace() = expression</code>
    * where the content's of matrix are overwritten inplace, even if the data is shared with
    * some other matrix instance.
    * Using the returned object in another context as directly as the left side of an assignment is
    * undefined behaviour.
    * The assignment will fail if the dimensions of this matrix and the expression don't match.
    * \return an expression to force inplace assignment
    */
    internal::SparseMatrixInplaceAssignment<Type> inplace()
    {
        return internal::MatrixInplaceAssignment<Type>(this);
    }
};

/**
* \brief Custom operator<< that prints the sparse matrix and additional information.
* First, information about the matrix like shape and storage options are printed,
* followed by the sparse data of the matrix.
*
* This operations involves copying the matrix from device to host.
* It is slow, use it only for debugging purpose.
* \param os the output stream
* \param m the matrix
* \return the output stream again
*/
template <typename _Scalar, int _Batches, int _SparseFlags>
__host__ std::ostream& operator<<(std::ostream& os, const SparseMatrix<_Scalar, _Batches, _SparseFlags>& m)
{
    os << "SparseMatrix: " << std::endl;
    os << " rows=" << m.rows();
    os << ", cols=" << m.cols();
    os << ", batches=" << m.batches() << " (" << (_Batches == Dynamic ? "dynamic" : "compile-time") << ")";
    os << ", storage=" << (_SparseFlags==SparseFlags::CSC ? "CSC" : "CSR") << std::endl;
    os << " Outer Indices (" << (_SparseFlags == SparseFlags::CSC ? "column" : "row") << "): " << m.getOuterIndices().toEigen().transpose() << std::endl;
    os << " Inner Indices (" << (_SparseFlags == SparseFlags::CSC ? "row" : "column") << "): " << m.getInnerIndices().toEigen().transpose() << std::endl;
    for (int batch = 0; batch < m.batches(); ++batch)
    {
        os << " Data (Batch " << batch << "): " << m.getData().slice(batch).eval().toEigen().transpose() << std::endl;
    }
    return os;
}

namespace internal
{
    template<typename _SparseMatrix>
    class SparseMatrixInplaceAssignment
    {
    private:
        _SparseMatrix * matrix_;
    public:
        SparseMatrixInplaceAssignment(_SparseMatrix* matrix) : matrix_(matrix) {}

        /**
        * \brief Evaluates the expression inline inplace into the current matrix.
        * No new memory is created, it is reused!
        * This operator fails with an exception if the dimensions don't match.
        * \param expr the other expression
        * \return the underlying matrix
        */
        template<typename Derived>
        CUMAT_STRONG_INLINE _SparseMatrix& operator=(const MatrixBase<Derived>& expr)
        {
            CUMAT_ASSERT_DIMENSION(matrix_->rows() == expr.rows());
            CUMAT_ASSERT_DIMENSION(matrix_->cols() == expr.cols());
            CUMAT_ASSERT_DIMENSION(matrix_->batches() == expr.batches());
            CUMAT_ASSERT(matrix_->isInitialized());
            Assignment<_SparseMatrix, Derived, AssignmentMode::ASSIGN, SparseDstTag, typename Derived::SrcTag>::assign(*matrix_, expr.derived());
            return *matrix_;
        }
    };
}

CUMAT_NAMESPACE_END


#endif