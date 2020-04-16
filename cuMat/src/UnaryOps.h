#ifndef __CUMAT_UNARY_OPS_H__
#define __CUMAT_UNARY_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "CwiseOp.h"
#include "NumTraits.h"

#include <cmath>
#include "../../third-party/cub/util_type.cuh"

CUMAT_NAMESPACE_BEGIN

namespace internal {
	template<typename _Child, typename _UnaryFunctor>
	struct traits<UnaryOp<_Child, _UnaryFunctor> >
	{
        //using Scalar = typename internal::traits<_Child>::Scalar;
        using Scalar = typename _UnaryFunctor::ReturnType;
		enum
		{
			Flags = internal::traits<_Child>::Flags,
			RowsAtCompileTime = internal::traits<_Child>::RowsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Child>::ColsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise
		};
        typedef CwiseSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
	};
}

/**
* \brief A generic unary operator.
* The unary functor can be any class or structure that supports
* the following method:
* \code
* __device__ const _Scalar& operator()(const Scalar& value, Index row, Index col, Index batch);
* \endcode
* \tparam _Child the matrix that is transformed by this unary function
* \tparam _UnaryFunctor the transformation functor
*/
template<typename _Child, typename _UnaryFunctor>
class UnaryOp : public CwiseOp<UnaryOp<_Child, _UnaryFunctor> >
{
public:
    typedef UnaryOp<_Child, _UnaryFunctor> Type;
	typedef CwiseOp<UnaryOp<_Child, _UnaryFunctor> > Base;
    CUMAT_PUBLIC_API

protected:
    typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
	const child_wrapped_t child_;
	const _UnaryFunctor functor_;

public:
	explicit UnaryOp(const MatrixBase<_Child>& child, const _UnaryFunctor& functor = _UnaryFunctor())
		: child_(child.derived()), functor_(functor)
	{}

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return child_.rows(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return child_.cols(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

	__device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
	{
		return functor_(child_.derived().coeff(row, col, batch, index), row, col, batch);
	}
};

// GENERAL UNARY OPERATIONS
namespace functor
{
#define DECLARE_FUNCTOR(Name) \
	template<typename _Scalar> class UnaryMathFunctor_ ## Name

#define DEFINE_GENERAL_FUNCTOR(Name, Fn) \
	template<typename _Scalar> \
	struct UnaryMathFunctor_ ## Name \
	{ \
	public: \
        typedef _Scalar ReturnType; \
		__device__ CUMAT_STRONG_INLINE _Scalar operator()(const _Scalar& x, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

#define DEFINE_FUNCTOR(Name, Scalar, Fn) \
	template<> \
	struct UnaryMathFunctor_ ## Name <Scalar> \
	{ \
	public: \
        typedef Scalar ReturnType; \
		__device__ CUMAT_STRONG_INLINE Scalar operator()(const Scalar& x, Index row, Index col, Index batch) const \
		{ \
			return Fn; \
		} \
	}

	//the std::is_same hack is needed so that the static assert evaluates to false only on template instantiation
#define DEFINE_FALLBACK_FUNCTOR(Name) \
	template<typename _Scalar> \
	struct UnaryMathFunctor_ ## Name \
	{ \
	public: \
		CUMAT_STATIC_ASSERT((!std::is_same<_Scalar, _Scalar>::value), "Functor not available for the selected type"); \
	}

#define DEFINE_FUNCTOR_FLOAT(Name, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);     \
	DEFINE_FUNCTOR(Name, float, Fn);   \
	DEFINE_FUNCTOR(Name, double, Fn);
#define DEFINE_FUNCTOR_FLOAT_COMPLEX(Name, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);     \
	DEFINE_FUNCTOR(Name, float, Fn);   \
	DEFINE_FUNCTOR(Name, double, Fn);  \
    DEFINE_FUNCTOR(Name, cfloat, Fn);  \
	DEFINE_FUNCTOR(Name, cdouble, Fn);
#define DEFINE_FUNCTOR_INT(Name, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);   \
	DEFINE_FUNCTOR(Name, int, Fn);   \
	DEFINE_FUNCTOR(Name, long, Fn);
#define DEFINE_FUNCTOR_SINGLE(Name, Scalar, Fn) \
	DEFINE_FALLBACK_FUNCTOR(Name);   \
	DEFINE_FUNCTOR(Name, Scalar, Fn);

	DEFINE_GENERAL_FUNCTOR(cwiseNegate, (-x));

	DEFINE_GENERAL_FUNCTOR(cwiseAbs, abs(x));
    template<>
    struct UnaryMathFunctor_cwiseAbs<cfloat>
    {
    public:
        typedef float ReturnType;
        __device__ CUMAT_STRONG_INLINE float operator()(const cfloat& x, Index row, Index col, Index batch) const
        {
            return abs(x);
        }
    };
    template<>
    struct UnaryMathFunctor_cwiseAbs<cdouble>
    {
    public:
        typedef double ReturnType;
        __device__ CUMAT_STRONG_INLINE double operator()(const cdouble& x, Index row, Index col, Index batch) const
        {
            return abs(x);
        }
    };

    DEFINE_GENERAL_FUNCTOR(cwiseAbs2, x*x);
    template<>
    struct UnaryMathFunctor_cwiseAbs2<cfloat>
    {
    public:
        typedef float ReturnType;
        __device__ CUMAT_STRONG_INLINE float operator()(const cfloat& x, Index row, Index col, Index batch) const
        {
            //using namespace CUMAT_NAMESPACE internal::complex_math;
            return norm(x);
        }
    };
    template<>
    struct UnaryMathFunctor_cwiseAbs2<cdouble>
    {
    public:
        typedef double ReturnType;
        __device__ CUMAT_STRONG_INLINE double operator()(const cdouble& x, Index row, Index col, Index batch) const
        {
            //using namespace CUMAT_NAMESPACE internal::complex_math;
            return norm(x);
        }
    };

	DEFINE_GENERAL_FUNCTOR(cwiseInverse, ( CUMAT_NAMESPACE internal::NumTraits<_Scalar>::RealType(1) / x ) );
    DEFINE_GENERAL_FUNCTOR(cwiseInverseCheck, ( x==_Scalar(0) ? _Scalar(1) : CUMAT_NAMESPACE internal::NumTraits<_Scalar>::RealType(1) / x ) );
    //DEFINE_CFUNCTOR(cwiseInverse, cfloat, (cfloat{ 1,0 } / x));
    //DEFINE_CFUNCTOR(cwiseInverse, cdouble, (cdouble{1,0} / x));

    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseExp, exp(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseLog, log(x));
	DEFINE_FUNCTOR_FLOAT(cwiseLog1p, log1p(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseLog10, log10(x));

    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseSqrt, sqrt(x));
	DEFINE_FUNCTOR_FLOAT(cwiseCbrt, cbrt(x));
#if CUMAT_NVCC==1
	DEFINE_FUNCTOR_FLOAT(cwiseRsqrt, rsqrt(x));
	DEFINE_FUNCTOR_FLOAT(cwiseRcbrt, rcbrt(x));
#else
	DEFINE_FUNCTOR_FLOAT(cwiseRsqrt, (x)); //Fallback to prevent the error that rsqrt and rcbrt are not found if not compiled with NVCC
	DEFINE_FUNCTOR_FLOAT(cwiseRcbrt, (x)); //An error is thrown when they are used either way by CUMAT_ERROR_IF_NO_NVCC
#endif
	
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseSin, sin(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseCos, cos(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseTan, tan(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseAsin, asin(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseAcos, acos(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseAtan, atan(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseSinh, sinh(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseCosh, cosh(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseTanh, tanh(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseAsinh, asinh(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseAcosh, acosh(x));
    DEFINE_FUNCTOR_FLOAT_COMPLEX(cwiseAtanh, atanh(x));

	DEFINE_FUNCTOR_FLOAT(cwiseFloor, floor(x));
	DEFINE_FUNCTOR_FLOAT(cwiseCeil, ceil(x));
	DEFINE_FUNCTOR_FLOAT(cwiseRound, round(x));
	DEFINE_FUNCTOR(cwiseFloor, int, x); DEFINE_FUNCTOR(cwiseFloor, long, x);;
	DEFINE_FUNCTOR(cwiseCeil, int, x); DEFINE_FUNCTOR(cwiseCeil, long, x);;
	DEFINE_FUNCTOR(cwiseRound, int, x); DEFINE_FUNCTOR(cwiseRound, long, x);;

	DEFINE_FUNCTOR_FLOAT(cwiseErf, erf(x));
	DEFINE_FUNCTOR_FLOAT(cwiseErfc, erfc(x));
	DEFINE_FUNCTOR_FLOAT(cwiseLgamma, lgamma(x));

    DEFINE_GENERAL_FUNCTOR(conjugate, x);
    DEFINE_FUNCTOR(conjugate, cfloat, conj(x));
    DEFINE_FUNCTOR(conjugate, cdouble, conj(x));

	DEFINE_FUNCTOR_INT(cwiseBinaryNot, ~x);
	DEFINE_FUNCTOR_SINGLE(cwiseLogicalNot, bool, !x);

#undef DECLARE_FUNCTOR
#undef DEFINE_GENERAL_FUNCTOR
#undef DEFINE_FALLBACK_FUNCTOR
#undef DEFINE_FUNCTOR
#undef DEFINE_FUNCTOR_FLOAT
#undef DEFINE_FUNCTOR_FLOAT_COMPLEX
#undef DEFINE_FUNCTOR_INT
#undef DEFINE_FUNCTOR_SINGLE
} //end namespace functor


// CASTING

namespace functor {
	/**
	 * \brief Cast a scalar of type _Source to a scalar of type _Target.
	 * The functor provides a function <code>static __device__ _Target cast(_Source source)</code>
	 * for casting.
	 * \tparam _Source the source type 
	 * \tparam _Target the target type
	 */
	template<typename _Source, typename _Target>
	struct CastFunctor
	{
		static __device__ CUMAT_STRONG_INLINE _Target cast(const _Source& source)
		{
			//general implementation
			return _Target(source);
		}
	};
	//TODO: specializations for complex
} //end namespace functor

namespace internal {
	template<typename _Child, typename _Target>
	struct traits<CastingOp<_Child, _Target> >
	{
		using Scalar = _Target;
		enum
		{
			Flags = internal::traits<_Child>::Flags,
			RowsAtCompileTime = internal::traits<_Child>::RowsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Child>::ColsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise
		};
        typedef CwiseSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
	};
}

/**
 * \brief Casting operator from the type of the matrix _Child to the datatype _Target.
 * It uses functor::CastFunctor for the casting
 * \tparam _Child the child expression
 * \tparam _Target the target type
 */
template<typename _Child, typename _Target>
class CastingOp : public CwiseOp<CastingOp<_Child, _Target> >
{
public:
	typedef CwiseOp<CastingOp<_Child, _Target> > Base;
    typedef CastingOp<_Child, _Target> Type;
    CUMAT_PUBLIC_API

	using SourceType = typename internal::traits<_Child>::Scalar;
	using TargetType = _Target;

protected:
    typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
    const child_wrapped_t child_;

public:
	explicit CastingOp(const MatrixBase<_Child>& child)
		: child_(child.derived())
	{}

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return child_.rows(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return child_.cols(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

	__device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
	{
		return functor::CastFunctor<SourceType, TargetType>::cast(child_.derived().coeff(row, col, batch, index));
	}
};


// diagonal() and asDiagonal()

namespace internal {

    /**
     * \brief Specialize this struct if you need a special conversion from the vector type to the matrix type.
     *  This is needed for example if your matrix uses some custom struct as small blocks as elements.
     * \tparam _VectorType the vector type
     */
    template<typename _VectorType>
    struct AsDiagonalFunctor
    {
        /**
         * \brief The matrix type after converting the vector entry to the matrix entry
         */
        using MatrixType = _VectorType;

        static __host__ __device__ CUMAT_STRONG_INLINE MatrixType asDiagonal(const _VectorType& v)
        {
            return v; //default implementation
        }
    };

    template<typename _Child>
    struct traits<AsDiagonalOp<_Child> >
    {
        using VectorType = typename internal::traits<_Child>::Scalar;
        using Scalar = typename AsDiagonalFunctor<VectorType>::MatrixType; //typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            Size = internal::traits<_Child>::RowsAtCompileTime == 1 ? internal::traits<_Child>::ColsAtCompileTime : internal::traits<_Child>::RowsAtCompileTime,
            RowsAtCompileTime = Size,
            ColsAtCompileTime = Size,
            BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise
        };
        typedef CwiseSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };
}
/**
* \brief A wrapper operation that expresses the compile-time vector as a diagonal matrix.
* This is the return type of \c asDiagonal()
* \tparam _Child the child expression
*/
template<typename _Child>
class AsDiagonalOp : public CwiseOp<AsDiagonalOp<_Child> >
{
public:
    typedef CwiseOp<AsDiagonalOp<_Child> > Base;
    typedef AsDiagonalOp<_Child> Type;
    CUMAT_PUBLIC_API
    using VectorType = typename internal::traits<Type>::VectorType;
    enum
    {
        Size = internal::traits<Type>::Size,
        IsRowVector = internal::traits<_Child>::RowsAtCompileTime == 1,
    };

protected:
    typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
    const child_wrapped_t child_;
    const Index size_;

public:
    explicit AsDiagonalOp(const MatrixBase<_Child>& child)
        : child_(child.derived())
        , size_(child.rows()==1 ? child.cols() : child.rows())
    {
        CUMAT_STATIC_ASSERT(internal::traits<_Child>::RowsAtCompileTime == 1 || internal::traits<_Child>::ColsAtCompileTime == 1,
            "The child expression must be a compile-time row or column vector");
    }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return size_; }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return size_; }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
    {
        using Functor = typename internal::AsDiagonalFunctor<VectorType>;
        if (row == col)
        {
            if (IsRowVector)
                return Functor::asDiagonal(child_.derived().coeff(0, col, batch, -1));
            else
                return Functor::asDiagonal(child_.derived().coeff(row, 0, batch, -1));
        } else
        {
            return typename Functor::MatrixType(0);
        }
    }
};


namespace internal {

    /**
     * \brief Specialize this struct if you need a special conversion from the matrix type to the vector type.
     *  This is needed for example if your matrix uses some custom struct as small blocks as elements.
     * \tparam _MatrixType the matrix type
     */
    template<typename _MatrixType>
    struct ExtractDiagonalFunctor
    {
        /**
         * \brief The matrix type after converting the vector entry to the matrix entry
         */
        using VectorType = _MatrixType;

        static __host__ __device__ CUMAT_STRONG_INLINE VectorType extractDiagonal(const _MatrixType& v)
        {
            return v; //default implementation
        }
    };

    template<typename _Child>
    struct traits<ExtractDiagonalOp<_Child> >
    {
        using MatrixType = typename internal::traits<_Child>::Scalar;
        using Scalar = typename ExtractDiagonalFunctor<MatrixType>::VectorType; //using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            RowsAtCompileTime = (internal::traits<_Child>::RowsAtCompileTime == Dynamic || internal::traits<_Child>::RowsAtCompileTime == Dynamic)
                ? Dynamic
                : (internal::traits<_Child>::ColsAtCompileTime < internal::traits<_Child>::RowsAtCompileTime
                    ? internal::traits<_Child>::ColsAtCompileTime
                    : internal::traits<_Child>::RowsAtCompileTime),
            ColsAtCompileTime = 1,
            BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise | WriteCwise
        };
        typedef CwiseSrcTag SrcTag;
        typedef DenseDstTag DstTag;
    };
}
/**
* \brief The operation that extracts the main diagonal of a matrix and returns it as a column vector.
* The matrix must not necessarily be square
* This is the return type of \c diagonal()
* \tparam _Child the child expression
*/
template<typename _Child>
class ExtractDiagonalOp : public CwiseOp<ExtractDiagonalOp<_Child> >
{
public:
    typedef CwiseOp<ExtractDiagonalOp<_Child> > Base;
    typedef ExtractDiagonalOp<_Child> Type;
    CUMAT_PUBLIC_API
    using MatrixType = typename internal::traits<Type>::MatrixType;

protected:
    typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
    const child_wrapped_t child_;
    const Index size_;

public:
    explicit ExtractDiagonalOp(const MatrixBase<_Child>& child)
        : child_(child.derived())
        , size_(child.rows() < child.cols() ? child.rows() : child.cols())
    {}

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return size_; }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return 1; }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return child_.batches(); }

    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
    {
        using Functor = internal::ExtractDiagonalFunctor<MatrixType>;
        return Functor::extractDiagonal(child_.derived().coeff(row, row, batch, index));
    }
};



// Extract Real+Imag part of a matrix

namespace internal {
    template <typename _Child, bool _Imag, bool _Lvalue>
    struct traits<ExtractComplexPartOp<_Child, _Imag, _Lvalue> >
    {
        typedef typename NumTraits<typename traits<_Child>::Scalar>::RealType Scalar;
        enum
        {
            Flags = traits<_Child>::Flags,
            RowsAtCompileTime = traits<_Child>::RowsAtCompileTime,
            ColsAtCompileTime = traits<_Child>::ColsAtCompileTime,
            BatchesAtCompileTime = traits<_Child>::BatchesAtCompileTime,
            AccessFlags = ReadCwise | WriteCwise
        };
        typedef CwiseSrcTag SrcTag;
        typedef DenseDstTag DstTag;
    };

} //end namespace internal

/**
 * \brief This operation extracts the real and the imaginary part of a complex matrix.
 * This is only available on complex matrices.
 * This is the RValue-version (const matrix)
 * \tparam _Child the child expression
 * \tparam _Imag true: imaginary part, false: real part
 */
template <typename _Child, bool _Imag>
class ExtractComplexPartOp<_Child, _Imag, false> : public CwiseOp<ExtractComplexPartOp<_Child, _Imag, false> >
{
public:
    using Base = CwiseOp<ExtractComplexPartOp<_Child, _Imag, false> >;
    using Type = ExtractComplexPartOp<_Child, _Imag, false>;
    CUMAT_PUBLIC_API

protected:
    typedef typename MatrixReadWrapper<_Child, ReadCwise>::type wrapped_matrix_t;
    const wrapped_matrix_t matrix_;

public:
    explicit ExtractComplexPartOp(const MatrixBase<_Child>& matrix)
        : matrix_(matrix.derived())
    {}

    /**
    * \brief Returns the number of rows of this matrix.
    * \return the number of rows
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return matrix_.rows(); }

    /**
    * \brief Returns the number of columns of this matrix.
    * \return the number of columns
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return matrix_.cols(); }

    /**
    * \brief Returns the number of batches of this matrix.
    * \return the number of batches
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }

    /**
    * \brief Accesses the coefficient at the specified coordinate for reading.
    * If the device supports it (CUMAT_ASSERT_CUDA is defined), the
    * access is checked for out-of-bound tests by assertions.
    * \param row the row index
    * \param col the column index
    * \param batch the batch index
    * \return a read-only reference to the entry
    */
    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
    {
        if (_Imag)
            return matrix_.coeff(row, col, batch, index).imag();
        else
            return matrix_.coeff(row, col, batch, index).real();
    }
};
/**
* \brief This operation extracts the real and the imaginary part of a complex matrix.
* This is only available on complex matrices.
* This is the LValue-version (non-const matrix)
* \tparam _Child the child expression
* \tparam _Imag true: imaginary part, false: real part
*/
template <typename _Child, bool _Imag>
class ExtractComplexPartOp<_Child, _Imag, true> : public CwiseOp<ExtractComplexPartOp<_Child, _Imag, true> >
{
public:
    using Base = CwiseOp<ExtractComplexPartOp<_Child, _Imag, true> >;
    using Type = ExtractComplexPartOp<_Child, _Imag, true>;
    CUMAT_PUBLIC_API

protected:
    _Child matrix_;

public:
    explicit ExtractComplexPartOp(MatrixBase<_Child>& matrix)
        : matrix_(matrix.derived())
    {
        CUMAT_STATIC_ASSERT(internal::traits<_Child>::AccessFlags & WriteCwise, 
            "Lvalue version of ExtractComplexPartOp must be called on a matrix expression with the WriteCwise-Flag");
    }

    /**
    * \brief Returns the number of rows of this matrix.
    * \return the number of rows
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return matrix_.rows(); }

    /**
    * \brief Returns the number of columns of this matrix.
    * \return the number of columns
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return matrix_.cols(); }

    /**
    * \brief Returns the number of batches of this matrix.
    * \return the number of batches
    */
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }

    /**
    * \brief Converts from the linear index back to row, column and batch index
    * \param index the linear index
    * \param row the row index (output)
    * \param col the column index (output)
    * \param batch the batch index (output)
    */
    __host__ __device__ CUMAT_STRONG_INLINE void index(Index index, Index& row, Index& col, Index& batch) const
    {
        matrix_.index(index, row, col, batch);
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
    __device__ CUMAT_STRONG_INLINE Scalar& coeff(Index row, Index col, Index batch, Index index)
    {
        if (_Imag)
            return matrix_.coeff(row, col, batch, index).imag();
        else
            return matrix_.coeff(row, col, batch, index).real();
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
    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
    {
        if (_Imag)
            return matrix_.coeff(row, col, batch, index).imag();
        else
            return matrix_.coeff(row, col, batch, index).real();
    }

    /**
    * \brief Access to the linearized coefficient.
    * The format of the indexing depends on whether this
    * matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
    * \param idx the linearized index of the entry.
    * \return the entry at that index
    */
    __device__ CUMAT_STRONG_INLINE void setRawCoeff(Index index, const Scalar& newValue)
    {
        auto val = matrix_.rawCoeff(index);
        if (_Imag)
            val.imag(newValue);
        else
            val.real(newValue);
        matrix_.setRawCoeff(index, val);
    }

    //ASSIGNMENT

    template<typename Derived>
    CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
    {
        CUMAT_ASSERT_ARGUMENT(rows() == expr.rows());
        CUMAT_ASSERT_ARGUMENT(cols() == expr.cols());
        CUMAT_ASSERT_ARGUMENT(batches() == expr.batches());
        internal::Assignment<Type, Derived, AssignmentMode::ASSIGN, internal::DenseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());
        return *this;
    }
};


// view-as
namespace internal
{
	template<typename _Child, Axis _Row, Axis _Col, Axis _Batch>
	struct traits<SwapAxisOp<_Child, _Row, _Col, _Batch> >
	{
		using Scalar = typename internal::traits<_Child>::Scalar;
		enum
		{
			Flags = internal::traits<_Child>::Flags,
			RowsAtCompileTime = 
				_Row == Axis::Row ? internal::traits<_Child>::RowsAtCompileTime
				: _Row == Axis::Column ? internal::traits<_Child>::ColsAtCompileTime
				: _Row == Axis::Batch ? internal::traits<_Child>::BatchesAtCompileTime
				: 1,
			ColsAtCompileTime =
				_Col == Axis::Row ? internal::traits<_Child>::RowsAtCompileTime
				: _Col == Axis::Column ? internal::traits<_Child>::ColsAtCompileTime
				: _Col == Axis::Batch ? internal::traits<_Child>::BatchesAtCompileTime
				: 1,
			BatchesAtCompileTime =
				_Batch == Axis::Row ? internal::traits<_Child>::RowsAtCompileTime
				: _Batch == Axis::Column ? internal::traits<_Child>::ColsAtCompileTime
				: _Batch == Axis::Batch ? internal::traits<_Child>::BatchesAtCompileTime
				: 1,
			AccessFlags = ReadCwise
		};
		typedef CwiseSrcTag SrcTag;
		typedef DeletedDstTag DstTag;
	};
}
/**
* \brief This operation allows to swap arbitrary axis.
* \tparam _Child the child expression
* \see MatrixBase::swapAxis()
*/
template<typename _Child, Axis _Row, Axis _Col, Axis _Batch>
class SwapAxisOp : public CwiseOp<SwapAxisOp<_Child, _Row, _Col, _Batch> >
{
	CUMAT_STATIC_ASSERT(_Row == Axis::NoAxis || _Row == Axis::Row || _Row == Axis::Column || _Row == Axis::Batch,
		"_Row parameter must be either NoAxis, Row, Column or Batch");
	CUMAT_STATIC_ASSERT(_Col == Axis::NoAxis || _Col == Axis::Row || _Col == Axis::Column || _Col == Axis::Batch,
		"_Col parameter must be either NoAxis, Row, Column or Batch");
	CUMAT_STATIC_ASSERT(_Batch == Axis::NoAxis || _Batch == Axis::Row || _Batch == Axis::Column || _Batch == Axis::Batch,
		"_Batch parameter must be either NoAxis, Row, Column or Batch");

public:
	typedef CwiseOp<SwapAxisOp<_Child, _Row, _Col, _Batch> > Base;
	typedef SwapAxisOp<_Child, _Row, _Col, _Batch> Type;
	CUMAT_PUBLIC_API

protected:
	typedef typename MatrixReadWrapper<_Child, AccessFlags::ReadCwise>::type child_wrapped_t;
	const child_wrapped_t child_;

public:
	explicit SwapAxisOp(const MatrixBase<_Child>& child)
		: child_(child.derived())
	{}

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const
	{
		return _Row == Axis::Row ? child_.rows()
			: _Row == Axis::Column ? child_.cols()
			: _Row == Axis::Batch ? child_.batches()
			: 1;
	}
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const
	{
		return _Col == Axis::Row ? child_.rows()
			: _Col == Axis::Column ? child_.cols()
			: _Col == Axis::Batch ? child_.batches()
			: 1;
	}
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const
	{
		return _Batch == Axis::Row ? child_.rows()
			: _Batch == Axis::Column ? child_.cols()
			: _Batch == Axis::Batch ? child_.batches()
			: 1;
	}

	__device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
	{
		return child_.derived().coeff(
			_Row == Axis::Row ? row
			: _Col == Axis::Row ? col
			: _Batch == Axis::Row ? batch
			: 0,
			_Row == Axis::Column ? row
			: _Col == Axis::Column ? col
			: _Batch == Axis::Column ? batch
			: 0,
			_Row == Axis::Batch ? row
			: _Col == Axis::Batch ? col
			: _Batch == Axis::Batch ? batch
			: 0,
			-1);
	}
};


CUMAT_NAMESPACE_END


//Global binary functions
CUMAT_FUNCTION_NAMESPACE_BEGIN

#define UNARY_OP(Name, Op) \
    template<typename _Derived> \
    CUMAT_NAMESPACE UnaryOp<_Derived, CUMAT_NAMESPACE functor::UnaryMathFunctor_ ## Op <typename CUMAT_NAMESPACE internal::traits<_Derived>::Scalar>> \
    Name(const CUMAT_NAMESPACE MatrixBase<_Derived>& mat) \
    { \
		CUMAT_ERROR_IF_NO_NVCC(Op)  \
        return CUMAT_NAMESPACE UnaryOp<_Derived, CUMAT_NAMESPACE functor::UnaryMathFunctor_ ## Op <typename CUMAT_NAMESPACE internal::traits<_Derived>::Scalar>>(mat.derived()); \
    }

UNARY_OP(abs, cwiseAbs);
UNARY_OP(inverse, cwiseInverse);
UNARY_OP(inverseCheck, cwiseInverseCheck);
UNARY_OP(floor, cwiseFloor);
UNARY_OP(ceil, cwiseCeil);
UNARY_OP(round, cwiseRound);

UNARY_OP(exp, cwiseExp);
UNARY_OP(log, cwiseLog);
UNARY_OP(log1p, cwiseLog1p);
UNARY_OP(log10, cwiseLog10);
UNARY_OP(sqrt, cwiseSqrt);
UNARY_OP(rsqrt, cwiseRsqrt);
UNARY_OP(cbrt, cwiseCbrt);
UNARY_OP(rcbrt, cwiseRcbrt);

UNARY_OP(sin, cwiseSin);
UNARY_OP(cos, cwiseCos);
UNARY_OP(tan, cwiseTan);
UNARY_OP(asin, cwiseAsin);
UNARY_OP(acos, cwiseAcos);
UNARY_OP(atan, cwiseAtan);
UNARY_OP(sinh, cwiseSinh);
UNARY_OP(cosh, cwiseCosh);
UNARY_OP(tanh, cwiseTanh);
UNARY_OP(asinh, cwiseAsinh);
UNARY_OP(acosh, cwiseAcosh);
UNARY_OP(atanh, cwiseAtanh);

UNARY_OP(erf, cwiseErf);
UNARY_OP(erfc, cwiseErfc);
UNARY_OP(lgamma, cwiseLgamma);

UNARY_OP(conjugate, conjugate);

template<typename _Child>
CUMAT_NAMESPACE ExtractComplexPartOp<_Child, false, false> real(const CUMAT_NAMESPACE MatrixBase<_Child>& mat)
{
	CUMAT_ERROR_IF_NO_NVCC(real)
    CUMAT_STATIC_ASSERT(CUMAT_NAMESPACE internal::NumTraits<typename CUMAT_NAMESPACE internal::traits<_Child>::Scalar>::IsComplex, "Matrix must be complex");
    return CUMAT_NAMESPACE ExtractComplexPartOp<_Child, false, false>(mat.derived());
}
template<typename _Child>
CUMAT_NAMESPACE ExtractComplexPartOp<_Child, true, false> imag(const CUMAT_NAMESPACE MatrixBase<_Child>& mat)
{
	CUMAT_ERROR_IF_NO_NVCC(imag)
    CUMAT_STATIC_ASSERT(CUMAT_NAMESPACE internal::NumTraits<typename CUMAT_NAMESPACE internal::traits<_Child>::Scalar>::IsComplex, "Matrix must be complex");
    return CUMAT_NAMESPACE ExtractComplexPartOp<_Child, true, false>(mat.derived());
}

#undef UNARY_OP

CUMAT_FUNCTION_NAMESPACE_END


#endif
