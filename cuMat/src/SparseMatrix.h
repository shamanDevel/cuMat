#ifndef __CUMAT_SPARSE_MATRIX__
#define __CUMAT_SPARSE_MATRIX__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "NumTraits.h"
#include "Constants.h"
#include "DevicePointer.h"
#include "MatrixBase.h"
#include "Matrix.h"
#include "SparseMatrixBase.h"

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
            SFlags = _SparseFlags,
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
class SparseMatrix : public SparseMatrixBase<SparseMatrix<_Scalar, _Batches, _SparseFlags> >
{
    CUMAT_STATIC_ASSERT(_SparseFlags == SparseFlags::CSR || _SparseFlags == SparseFlags::CSC,
        "SparseFlags must be either CSR or CSC");
public:

    using Type = SparseMatrix<_Scalar, _Batches, _SparseFlags>;
    typedef SparseMatrixBase<SparseMatrix<_Scalar, _Batches, _SparseFlags> > Base;
    CUMAT_PUBLIC_API_NO_METHODS
    using Base::derived;
    enum
    {
        SFlags = _SparseFlags
    };

    using typename Base::StorageIndex;
    using typename Base::IndexVector;
    using typename Base::ConstIndexVector;

    typedef Matrix<Scalar, Dynamic, 1, Batches, Flags::ColumnMajor> ScalarVector;
    typedef const Matrix<Scalar, Dynamic, 1, Batches, Flags::ColumnMajor> ConstScalarVector;

private:
    /**
     * \brief The (possibly batched) vector with the coefficients of size nnz_ .
     */
    ScalarVector A_;

public:

    //----------------------------------
    //  CONSTRUCTORS
    //----------------------------------

    /**
     * \brief Default constructor, SparseMatrix is empty
     */
    SparseMatrix() = default;

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
        : Base(sparsityPattern, batches)
        , A_(sparsityPattern.nnz, 1, batches) //This also checks if the number of batches is valid
    {
    }

    __host__ __device__ CUMAT_STRONG_INLINE ScalarVector& getData() { return A_; }
    __host__ __device__ CUMAT_STRONG_INLINE const ConstScalarVector& getData() const { return A_; }
    using Base::getInnerIndices;
    using Base::getOuterIndices;
    using Base::isInitialized;

    //----------------------------------
    //  COEFFICIENT ACCESS
    //----------------------------------

    using Base::rows;
    using Base::cols;
    using Base::batches;
    using Base::nnz;
    using Base::size;
    using Base::outerSize;
    using Base::IA_;
    using Base::JA_;

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
            const int start = JA_.getRawCoeff(col);
            const int end = JA_.getRawCoeff(col + 1);
            for (int i=start; i<end; ++i)
            {
                int r = IA_.getRawCoeff(i);
                if (r == row) return A_.coeff(i, 0, batch, -1);
            }
        } else //CSR
        {
            const int start = JA_.getRawCoeff(row);
            const int end = JA_.getRawCoeff(row + 1);
            for (int i = start; i<end; ++i)
            {
                int c = IA_.getRawCoeff(i);
                if (c == col) return A_.coeff(i, 0, batch, -1);
            }
        }
        return Scalar(0);
    }

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

    /**
     * \brief Implementation of \ref SparseMatrixBase::getSparseCoeff(Index row, Index col, Index batch, Index index) const.
     * simply passes on the index to getRawCoeff(Index)
     * \param index 
     * \return 
     */
    __device__ CUMAT_STRONG_INLINE const _Scalar& getSparseCoeff(Index /*row*/, Index /*col*/, Index /*batch*/, Index index) const
    {
        return getRawCoeff(index);
    }

    /**
     * \brief Returns a wrapper class for direct coefficient access.
     * If this wrapper is used, a call to \ref coeff(Index row, Index col, Index batch, Index linear) const
     * will use the linear index instead of the regular row-column-batch index. This avoids
     * the search for the matching coefficient in this sparse matrix.
     * 
     * The use case is the following optimization:
     * Assume the sparse matrices A and B have the same sparsity pattern (and same storage order)
     * and the following operation
     * \code A = cwiseop(B) \endcode with \c cwiseop() being any chain of compouned-wise operations
     * (unary, binary) that does not change the access order (no broadcasting, transposition, ...).
     * Then the order in which the entries in B are accessed are exactly the same as how the elements into A 
     * are written.
     * This is a very common case and in that scenario, the search for the current entry in B can be
     * avoided by using directly the raw index, that is used to write into A, also to read the entry in B.
     * This optimization has to be manually triggered by the user by calling this method on B:
     * \code A = cwiseop(B.direct()) \endcode .
     * 
     * If the sparsity pattern of this matrix and the target matrix (A above) do no match,
     * the results of the evaluation are unspecified.
     * 
     * \return a wrapper to trigger the direct-read optimization
     */
    internal::SparseMatrixDirectAccess<const Type> direct() const
    {
        return internal::SparseMatrixDirectAccess<const Type>(this);
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
     * \brief Performs a deep clone of the matrix.
     * Usually, when you assign matrix instances, the underlying data is shared. This method explicitly copies the internal data
     * of the matrix into a new matrix. The two matrices are then completely independent, i.e. if you perform an 
     * inplace modification on this or the returned matrix, the changes are not reflected in the other.
     * 
     * You can select whether only the data (\code cloneSparsity=false \endcode, the default) is cloned,
     * or also the sparsity pattern (\code cloneSparsity=true \endcode).
     *
     * \param cloneSparsity false -> only the data array is cloned, true -> data array and sparsity pattern is cloned
     * \return the new matrix with a deep clone of the data
     */
    Type deepClone(bool cloneSparsity = false) const
	{
        Type mat(*this); //shallow
        mat.A_ = mat.A_.deepClone();
        if (cloneSparsity)
        {
            mat.IA_ = mat.IA_.deepClone();
            mat.JA_ = mat.JA_.deepClone();
        }
        return mat;
	}

private:
    void checkInitialized() const
    {
        if (!isInitialized())
        {
            throw std::runtime_error("The sparsity pattern of this SparseMatrix has not been initialized, can't assign to this matrix");
        }
    }
public:

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
        checkInitialized();
        makeExclusiveUse();
        internal::Assignment<Type, Derived, AssignmentMode::ASSIGN, internal::SparseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());
        return *this;
    }
    //No evaluation constructor

    #define CUMAT_COMPOUND_ASSIGNMENT(op, mode)                                                         \
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
        /*expr.template evalTo<Type, AssignmentMode:: mode >(*this);*/                                  \
        checkInitialized();                                                                             \
        internal::Assignment<Type, Derived, AssignmentMode:: mode , internal::SparseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());   \
        return *this;                                                                                   \
    }

    CUMAT_COMPOUND_ASSIGNMENT(operator+=, ADD)
    CUMAT_COMPOUND_ASSIGNMENT(operator-=, SUB)
    //CUMAT_COMPOUND_ASSIGNMENT(operator*=, MUL) //multiplication is ambigious: do you want cwise or matrix multiplication?
    CUMAT_COMPOUND_ASSIGNMENT(operator/=, DIV)
    CUMAT_COMPOUND_ASSIGNMENT(operator%=, MOD)
    CUMAT_COMPOUND_ASSIGNMENT(operator&=, AND)
    CUMAT_COMPOUND_ASSIGNMENT(operator|=, OR)

#undef CUMAT_COMPOUND_ASSIGNMENT

    /**
     * \brief Explicit overloading of \c operator*= for scalar right hand sides.
     * This is needed to disambiguate the difference between component-wise operations and matrix operations.
     * All other compount-assignment operators (+=, -=, /=, ...) act component-wise.
     * 
     * The operator *= is special: if call with a scalar argument, it simply scales the argument;
     * if called with a matrix as argument, it performs an inplace matrix multiplication.
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
        internal::Assignment<Type, Expr, AssignmentMode::MUL, internal::SparseDstTag, typename internal::traits<Expr>::SrcTag>::assign(*this, Type::Constant(rows(), cols(), batches(), scalar));
        return *this;
	}

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
        return internal::SparseMatrixInplaceAssignment<Type>(this);
    }

    // STATIC METHODS AND OTHER HELPERS
    /**
    * \brief Sets all entries to zero.
    * Warning: this operation works in-place and therefore violates the copy-on-write paradigm.
    */
    void setZero()
    {
        A_.setZero();
    }

#include "MatrixNullaryOpsPlugin.inl"
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

    template<typename _SparseMatrix>
    struct traits<SparseMatrixDirectAccess<_SparseMatrix> >
    {
        using Scalar = typename traits<_SparseMatrix>::Scalar;
        enum
        {
            Flags = ColumnMajor, //always use ColumnMajor when evaluated to dense stuff
            SFlags = traits<_SparseMatrix>::SFlags,
            RowsAtCompileTime = Dynamic,
            ColsAtCompileTime = Dynamic,
            BatchesAtCompileTime = traits<_SparseMatrix>::BatchesAtCompileTime,
            AccessFlags = ReadCwise
        };
        typedef CwiseSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };

    template<typename _SparseMatrix>
    class SparseMatrixDirectAccess : public MatrixBase<SparseMatrixDirectAccess<_SparseMatrix> >
    {
    public:
        using Type = SparseMatrixDirectAccess<_SparseMatrix>;
        using Base = MatrixBase<Type>;
        CUMAT_PUBLIC_API

    private:
        _SparseMatrix matrix_;
    public:
        SparseMatrixDirectAccess(const _SparseMatrix* matrix) : matrix_(*matrix) {} //store by value, needed for the host->device transfer

        __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return matrix_.rows(); }
        __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return matrix_.cols(); }
        __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }
        /**
        * \brief Accesses a single entry, used directly the linear index to access the entry.
        * \param linear the linear index
        * \return the scalar coefficient at this position
        * \see SparseMatrix::direct()
        */
        __device__ const Scalar& coeff(Index /*row*/, Index /*col*/, Index /*batch*/, Index linear) const
        {
            return matrix_.getRawCoeff(linear);
        }
    };
} //end namespace internal

CUMAT_NAMESPACE_END


#endif