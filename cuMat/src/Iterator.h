#ifndef __CUMAT_ITERATOR_H__
#define __CUMAT_ITERATOR_H__

#include "Macros.h"
#include "ForwardDeclarations.h"

#include <array>

CUMAT_NAMESPACE_BEGIN

/**
 * \brief A random-access matrix iterator with adaptive stride.
 * This is a very general iterator that allows the traversal of a matrix in any order
 * (row-column-batch, or batch-column-row, to name a few).
 * Parts of the code are taken from CUB.
 * \tparam _Derived 
 */
template <typename _Derived>
class StridedMatrixIterator
{
public:
    // Required iterator traits
    typedef StridedMatrixIterator<_Derived> self_type; ///< My own type
    typedef Index difference_type; ///< Type to express the result of subtracting one iterator from another
    using ValueType = typename internal::traits<_Derived>::Scalar;
    typedef ValueType value_type; ///< The type of the element the iterator can point to
    typedef ValueType* pointer; ///< The type of a pointer to an element the iterator can point to
    typedef ValueType reference; ///< The type of a reference to an element the iterator can point to

protected:
    _Derived mat_;
    std::array<Index, 3> dims_;
    std::array<Index, 3> stride_;
    Index index_;

public:
    /// Constructor
    __host__ __device__
    StridedMatrixIterator(const MatrixBase<_Derived>& mat, std::array<Index, 3> stride)
        : mat_(mat.derived())
        , dims_{mat.rows(), mat.cols(), mat.batches()}
        , stride_(stride)
        , index_(0)
    {}

    __host__ __device__
    static Index toLinear(const std::array<Index, 3> coords, const std::array<Index, 3> stride)
    {
        Index l = 0;
        //for (int i = 0; i < 3; ++i) l += coords[i] * stride[i];
        //manual loop unrolling
        l += coords[0] * stride[0];
        l += coords[1] * stride[1];
        l += coords[2] * stride[2];
        return l;
    }

    __host__ __device__
    static std::array<Index, 3> fromLinear(Index linear, const std::array<Index, 3>& dims, const std::array<Index, 3>& stride)
    {
        //for (int i = 0; i < 3; ++i) outCoords[i] = (linear / stride[i]) % dims[i];
        //manual loop unrolling
        return {
            (linear / stride[0]) % dims[0],
            (linear / stride[1]) % dims[1],
            (linear / stride[2]) % dims[2]
        };
    }

    /// Postfix increment
    __host__ __device__ CUMAT_STRONG_INLINE self_type operator++(int)
    {
        self_type retval = *this;
        index_++;
        return retval;
    }

    /// Prefix increment
    __host__ __device__ CUMAT_STRONG_INLINE self_type operator++()
    {
        index_++;
        return *this;
    }

    /// Indirection
    __device__ CUMAT_STRONG_INLINE reference operator*() const
    {
        std::array<Index, 3> coords = fromLinear(index_, dims_, stride_);
        return mat_.coeff(coords[0], coords[1], coords[2]);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ CUMAT_STRONG_INLINE self_type operator+(Distance n) const
    {
        self_type retval = *this;
        retval.index_ += n;
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ CUMAT_STRONG_INLINE self_type& operator+=(Distance n)
    {
        index_ += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ CUMAT_STRONG_INLINE self_type operator-(Distance n) const
    {
        self_type retval = *this;
        retval.index_ -= n;
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ CUMAT_STRONG_INLINE self_type& operator-=(Distance n)
    {
        index_ -= n;
        return *this;
    }

    /// Distance
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return index_ - other.index_;
    }

    /// Array subscript
    template <typename Distance>
    __device__ __forceinline__ reference operator[](Distance n) const
    {
        std::array<Index, 3> coords = fromLinear(index_ + n, dims_, stride_);
        return mat_.coeff(coords[0], coords[1], coords[2]);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (index_ == rhs.index_);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (index_ != rhs.index_);
    }
};


/**
* \brief A random-access input generator for dereferencing a sequence of incrementing integer values.
* This is an extension to the CountingInputIterator from CUB to specify the increment.
*/
template <
    typename ValueType = Index,
    typename OffsetT = Index>
    class CountingInputIterator
{
public:

    // Required iterator traits
    typedef CountingInputIterator               self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to

private:

    ValueType val;
    ValueType increment;

public:

    /// Constructor
    __host__ __device__ __forceinline__ CountingInputIterator(
        const ValueType &val,          ///< Starting value for the iterator instance to report
        const ValueType &increment=1)  ///< The increment
        : val(val), increment(increment)
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        ++val;
        return retval;
    }

    /// Prefix increment
    __host__ __device__ __forceinline__ self_type operator++()
    {
        ++val;
        return *this;
    }

    /// Indirection
    __host__ __device__ __forceinline__ reference operator*() const
    {
        return val*increment;
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(val + ValueType(n), increment);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        val += ValueType(n);
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(val - ValueType(n), increment);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        val -= n;
        return *this;
    }

    /// Distance
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return difference_type(val - other.val);
    }

    /// Array subscript
    template <typename Distance>
    __host__ __device__ __forceinline__ reference operator[](Distance n) const
    {
        return (val + ValueType(n)) * increment;
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        os << "[" << itr.val*itr.increment << "]";
        return os;
    }

};

CUMAT_NAMESPACE_END

#endif