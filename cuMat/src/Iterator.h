#ifndef __CUMAT_ITERATOR_H__
#define __CUMAT_ITERATOR_H__

#include "Macros.h"
#include "ForwardDeclarations.h"

#include <array>
#include <iterator>
#include <thrust/tuple.h>

CUMAT_NAMESPACE_BEGIN

/**
 * \brief A random-access matrix input iterator with adaptive stride.
 * This is a very general iterator that allows the traversal of a matrix in any order
 * (row-column-batch, or batch-column-row, to name a few).
 * Parts of the code are taken from CUB.
 * \tparam _Derived 
 */
template <typename _Derived>
class StridedMatrixInputIterator
{
public:
    // Required iterator traits
    typedef StridedMatrixInputIterator<_Derived> self_type; ///< My own type
    typedef Index difference_type; ///< Type to express the result of subtracting one iterator from another
    using ValueType = typename internal::traits<_Derived>::Scalar;
    using value_type = ValueType; ///< The type of the element the iterator can point to
    using pointer = ValueType*; ///< The type of a pointer to an element the iterator can point to
    using reference = ValueType&; ///< The type of a reference to an element the iterator can point to
    using iterator_category = std::random_access_iterator_tag;

    typedef thrust::tuple<Index, Index, Index> Index3;

protected:
    _Derived mat_;
    Index3 dims_;
    Index3 stride_;
    Index index_;

public:
    /// Constructor
    __host__ __device__
    StridedMatrixInputIterator(const MatrixBase<_Derived>& mat, const Index3& stride)
        : mat_(mat.derived())
        , dims_{mat.rows(), mat.cols(), mat.batches()}
        , stride_(stride)
        , index_(0)
    {}

    __host__ __device__
    static Index toLinear(const Index3& coords, const Index3& stride)
    {
        Index l = 0;
        //for (int i = 0; i < 3; ++i) l += coords[i] * stride[i];
        //manual loop unrolling
        l += coords.get<0>() * stride.get<0>();
        l += coords.get<1>() * stride.get<1>();
        l += coords.get<2>() * stride.get<2>();
        return l;
    }

    __host__ __device__
    static Index3 fromLinear(Index linear, const Index3& dims, const Index3& stride)
    {
        //for (int i = 0; i < 3; ++i) outCoords[i] = (linear / stride[i]) % dims[i];
        //manual loop unrolling
        Index3 coords = {
            (linear / stride.get<0>()) % dims.get<0>(),
            (linear / stride.get<1>()) % dims.get<1>(),
            (linear / stride.get<2>()) % dims.get<2>()
        };
        //printf("index: %d; stride: %d,%d,%d  -> coords: %d,%d,%d\n",
        //    (int)linear,
        //    (int)stride.get<0>(), (int)stride.get<1>(), (int)stride.get<2>(),
        //    (int)coords.get<0>(), (int)coords.get<1>(), (int)coords.get<2>());
        return coords;
    }

    /// Postfix increment
    __host__ __device__ CUMAT_STRONG_INLINE self_type operator++(int)
    {
        self_type retval = *this;
        index_++;
        return retval;
    }

    /// Prefix increment
    __host__ __device__ CUMAT_STRONG_INLINE self_type& operator++()
    {
        index_++;
        return *this;
    }

    /// Indirection
    __device__ CUMAT_STRONG_INLINE value_type operator*() const
    {
        Index3 coords = fromLinear(index_, dims_, stride_);
        return mat_.coeff(coords.get<0>(), coords.get<1>(), coords.get<2>(), -1);
    }

    __device__ CUMAT_STRONG_INLINE reference operator*()
    {
        Index3 coords = fromLinear(index_, dims_, stride_);
        return mat_.coeff(coords.get<0>(), coords.get<1>(), coords.get<2>(), -1);
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
    __device__ __forceinline__ value_type operator[](Distance n) const
    {
        Index3 coords = fromLinear(index_+n, dims_, stride_);
        return mat_.coeff(coords.get<0>(), coords.get<1>(), coords.get<2>(), -1);
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
* \brief A random-access matrix input iterator with adaptive stride.
* This is a very general iterator that allows the traversal of a matrix in any order
* (row-column-batch, or batch-column-row, to name a few).
* Parts of the code are taken from CUB.
* \tparam _Derived
*/
template <typename _Derived>
class StridedMatrixOutputIterator : public StridedMatrixInputIterator<_Derived>
{
public:
    // Required iterator traits
    typedef StridedMatrixInputIterator<_Derived> Base;
    typedef StridedMatrixOutputIterator<_Derived> self_type; ///< My own type
    typedef Index difference_type; ///< Type to express the result of subtracting one iterator from another
    using ValueType = typename internal::traits<_Derived>::Scalar;
    using value_type = ValueType; ///< The type of the element the iterator can point to
    using pointer = ValueType*; ///< The type of a pointer to an element the iterator can point to
    using reference = ValueType&; ///< The type of a reference to an element the iterator can point to
    using iterator_category = std::random_access_iterator_tag;

    typedef thrust::tuple<Index, Index, Index> Index3;

public:
    /// Constructor
    __host__ __device__
        StridedMatrixOutputIterator(const MatrixBase<_Derived>& mat, const Index3& stride)
        : StridedMatrixInputIterator<_Derived>(mat, stride)
    {}

    //The only new thing is the non-const version of the dereference

    template <typename Distance>
    __device__ __forceinline__ reference operator[](Distance n)
    {
        Index3 coords = fromLinear(Base::index_ + n, Base::dims_, Base::stride_);
        return Base::mat_.coeff(coords.get<0>(), coords.get<1>(), coords.get<2>(), -1);
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
    using iterator_category = std::random_access_iterator_tag;

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