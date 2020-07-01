#ifndef __CUMAT_MATRIX_H__
#define __CUMAT_MATRIX_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Errors.h"
#include "NumTraits.h"
#include "CudaUtils.h"
#include "Constants.h"
#include "Context.h"
#include "DevicePointer.h"
#include "MatrixBase.h"
#include "CwiseOp.h"
#include "NullaryOps.h"
#include "TransposeOp.h"
#include "IO.h"

#if CUMAT_EIGEN_SUPPORT==1
#include <Eigen/Core>
#include "EigenInteropHelpers.h"
#endif

CUMAT_NAMESPACE_BEGIN

namespace internal {

	/**
	 * \brief The storage for the matrix
	 */
	template <typename _Scalar, int _Rows, int _Columns, int _Batches>
	class DenseStorage;

	//purely fixed size
	template <typename _Scalar, int _Rows, int _Columns, int _Batches>
	class DenseStorage
	{
		DevicePointer<_Scalar> data_;
	public:
		DenseStorage() : data_(Index(_Rows) * _Columns * _Batches) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_) {}
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) data_ = other.data_;
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_(Index(_Rows) * _Columns * _Batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows && cols == _Columns && batches == _Batches);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows && cols == _Columns && batches == _Batches);
		}
		void swap(DenseStorage& other) { std::swap(data_, other.data_); }
		static __host__ __device__ CUMAT_STRONG_INLINE Index rows() { return _Rows; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index cols() { return _Columns; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index batches() { return _Batches; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//TODO: do I need specializations for null-matrices?

	//partly dynamic size

	//dynamic number of rows
	template <typename _Scalar, int _Columns, int _Batches>
	class DenseStorage<_Scalar, Dynamic, _Columns, _Batches>
	{
		DevicePointer<_Scalar> data_;
		Index rows_;
	public:
		DenseStorage() : data_(), rows_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_), rows_(other.rows_) {}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				rows_ = other.rows_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_((rows>=0?rows:0) * _Columns * _Batches)
			, rows_(rows)
		{
			CUMAT_ASSERT_ARGUMENT(cols == _Columns && batches == _Batches);
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, rows_(rows)
		{
			CUMAT_ASSERT_ARGUMENT(cols == _Columns && batches == _Batches);
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(rows_, other.rows_);
		}
		__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index cols() { return _Columns; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index batches() { return _Batches; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//dynamic number of cols
	template <typename _Scalar, int _Rows, int _Batches>
	class DenseStorage<_Scalar, _Rows, Dynamic, _Batches>
	{
		DevicePointer<_Scalar> data_;
		Index cols_;
	public:
		DenseStorage() : data_(), cols_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_), cols_(other.cols_) {}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				cols_ = other.cols_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_(_Rows * (cols>=0?cols:0) * _Batches)
			, cols_(cols)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows && batches == _Batches);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, cols_(cols)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows && batches == _Batches);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(cols_, other.cols_);
		}
		static __host__ __device__ CUMAT_STRONG_INLINE Index rows() { return _Rows; }
		__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index batches() { return _Batches; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//dynamic number of batches
	template <typename _Scalar, int _Rows, int _Columns>
	class DenseStorage<_Scalar, _Rows, _Columns, Dynamic>
	{
		DevicePointer<_Scalar> data_;
		Index batches_;
	public:
		DenseStorage() : data_(), batches_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_), batches_(other.batches_) {}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				batches_ = other.batches_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_(Index(_Rows) * _Columns * (batches>=0?batches:0))
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows && cols == _Columns);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows && cols == _Columns);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(batches_, other.batches_);
		}
		static __host__ __device__ CUMAT_STRONG_INLINE Index rows() { return _Rows; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index cols() { return _Columns; }
		__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//dynamic number of rows and cols
	template <typename _Scalar, int _Batches>
	class DenseStorage<_Scalar, Dynamic, Dynamic, _Batches>
	{
		DevicePointer<_Scalar> data_;
		Index rows_;
		Index cols_;
	public:
		DenseStorage() : data_(), rows_(0), cols_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				rows_ = other.rows_;
				cols_ = other.cols_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_((rows>=0?rows:0) * (cols>=0?cols:0) * _Batches)
			, rows_(rows)
			, cols_(cols)
		{
			CUMAT_ASSERT_ARGUMENT(batches == _Batches);
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, rows_(rows)
			, cols_(cols)
		{
			CUMAT_ASSERT_ARGUMENT(batches == _Batches);
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(rows_, other.rows_);
			std::swap(cols_, other.cols_);
		}
		__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }
		__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index batches() { return _Batches; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//dynamic number of rows and batches
	template <typename _Scalar, int _Columns>
	class DenseStorage<_Scalar, Dynamic, _Columns, Dynamic>
	{
		DevicePointer<_Scalar> data_;
		Index rows_;
		Index batches_;
	public:
		DenseStorage() : data_(), rows_(0), batches_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_), rows_(other.rows_), batches_(other.batches_) {}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				rows_ = other.rows_;
				batches_ = other.batches_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_((rows>=0?rows:0) * _Columns * (batches>=0?batches:0))
			, rows_(rows)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(cols == _Columns);
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, rows_(rows)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(cols == _Columns);
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(rows_, other.rows_);
			std::swap(batches_, other.batches_);
		}
		__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index cols() { return _Columns; }
		__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//dynamic number of cols and batches
	template <typename _Scalar, int _Rows>
	class DenseStorage<_Scalar, _Rows, Dynamic, Dynamic>
	{
		DevicePointer<_Scalar> data_;
		Index cols_;
		Index batches_;
	public:
		DenseStorage() : data_(), cols_(0), batches_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) : data_(other.data_), cols_(other.cols_), batches_(other.batches_) {}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				cols_ = other.cols_;
				batches_ = other.batches_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_(Index(_Rows) * (cols>=0?cols:0) * (batches>=0?batches:0))
			, cols_(cols)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, cols_(cols)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows == _Rows);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(cols_, other.cols_);
			std::swap(batches_, other.batches_);
		}
		static __host__ __device__ CUMAT_STRONG_INLINE Index rows() { return _Rows; }
		__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_; }
		__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	//everything is dynamic
	template <typename _Scalar>
	class DenseStorage<_Scalar, Dynamic, Dynamic, Dynamic>
	{
		DevicePointer<_Scalar> data_;
		Index rows_;
		Index cols_;
		Index batches_;
	public:
		DenseStorage() : data_(), rows_(0), cols_(0), batches_(0) {}
        __host__ __device__
		DenseStorage(const DenseStorage& other) 
			: data_(other.data_)
			, rows_(other.rows_)
			, cols_(other.cols_)
			, batches_(other.batches_)
		{}
        __host__ __device__
		DenseStorage& operator=(const DenseStorage& other)
		{
			if (this != &other) {
				data_ = other.data_;
				rows_ = other.rows_;
				cols_ = other.cols_;
				batches_ = other.batches_;
			}
			return *this;
		}
		DenseStorage(Index rows, Index cols, Index batches)
			: data_((rows>=0?rows:0) * (cols>=0?cols:0) * (batches>=0?batches:0))
			, rows_(rows)
			, cols_(cols)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		DenseStorage(const DevicePointer<_Scalar>& data, Index rows, Index cols, Index batches)
			: data_(data)
			, rows_(rows)
			, cols_(cols)
			, batches_(batches)
		{
			CUMAT_ASSERT_ARGUMENT(rows >= 0);
			CUMAT_ASSERT_ARGUMENT(cols >= 0);
			CUMAT_ASSERT_ARGUMENT(batches >= 0);
		}
		void swap(DenseStorage& other) noexcept
		{
			std::swap(data_, other.data_);
			std::swap(rows_, other.rows_);
			std::swap(cols_, other.cols_);
			std::swap(batches_, other.batches_);
		}
		__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return rows_; }
		__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return cols_; }
		__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return batches_; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
		CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const { return data_; }
		CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer() { return data_; }
	};

	template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
	struct traits<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> >
	{
		typedef _Scalar Scalar;
		enum
		{
			Flags = _Flags,
			RowsAtCompileTime = _Rows,
			ColsAtCompileTime = _Columns,
			BatchesAtCompileTime = _Batches,
            AccessFlags = ReadCwise | ReadDirect | WriteCwise | WriteDirect | RWCwise | RWCwiseRef
		};
        typedef CwiseSrcTag SrcTag;
        typedef DenseDstTag DstTag;
	};

} //end namespace internal

/**
 * \brief The basic matrix class.
 * It is used to store batched matrices and vectors of
 * compile-time constant size or dynamic size.
 * 
 * For the sake of performance, set as many dimensions to compile-time constants as possible.
 * This allows to choose the best algorithm already during compilation.
 * There is no limit on the size of the compile-time dimensions, since all memory lives in 
 * the GPU memory, not on the stack (as opposed to Eigen).
 * 
 * The matrix class is a very slim class. It follows the copy-on-write principle.
 * This means that all copies of the matrices (created on assignment) share the same
 * underlying memory. Only if the contents are changed, the changes are written into
 * new memory (or in the same if this matrix uses the underlying memory exlusivly).
 * 
 * \tparam _Scalar the scalar type of the matrix
 * \tparam _Rows the number of rows, can be a compile-time constant or cuMat::Dynamic
 * \tparam _Columns the number of cols, can be a compile-time constant or Dynamic
 * \tparam _Batches the number of batches, can be a compile-time constant or Dynamic
 * \tparam _Flags a combination of flags from the \ref Flags enum.
 */
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
class Matrix : public CwiseOp<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> > 
    //inheriting from CwiseOp instead MatrixBase allows it to be evaluated as cwise-operation into lvalues.
{
protected:
	using Storage_t = internal::DenseStorage<_Scalar, _Rows, _Columns, _Batches>;
	Storage_t data_;
public:
	
	typedef Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> Type;
	typedef CwiseOp<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> > Base;
    CUMAT_PUBLIC_API
    enum
    {
        TransposedFlags = CUMAT_IS_COLUMN_MAJOR(Flags) ? RowMajor : ColumnMajor
    };
	using Base::size;

    typedef Matrix<const _Scalar, _Rows, _Columns, _Batches, _Flags> ConstType;
    typedef Matrix<typename std::remove_const<_Scalar>::type, _Rows, _Columns, _Batches, _Flags> NonConstType;

	/**
	 * \brief Default constructor.
	 * For completely fixed-size matrices, this creates a matrix of that size.
	 * For (fully or partially) dynamic matrices, creates a matrix of size 0.
	 */
    __host__
	Matrix() {}

#ifdef CUMAT_PARSED_BY_DOXYGEN
	/**
	* \brief Creates a vector (row or column) of the specified size.
	* This constructor is only allowed for compile-time row or column vectors with one batch and dynamic size.
	* \param size the size of the vector
	*/
	explicit Matrix(Index size) {}
#else
	template<typename T = std::enable_if<(_Rows == 1 && _Columns == Dynamic && _Batches == 1) || (_Columns == 1 && _Rows == Dynamic && _Batches == 1), Index>>
	explicit Matrix(typename T::type size)
		: data_(_Rows==1 ? 1 : size, _Columns==1 ? 1 : size, 1)
	{}
#endif

#ifdef CUMAT_PARSED_BY_DOXYGEN
	/**
	* \brief Creates a matrix of the specified size.
	* This constructor is only allowed for matrices with a batch size of one during compile time.
	* \param rows the number of rows
	* \param cols the number of batches
	*/
	Matrix(Index rows, Index cols) {}
#else
	template<typename T = std::enable_if<_Batches == 1, Index>>
	Matrix(typename T::type rows, Index cols)
		: data_(rows, cols, 1)
	{}
#endif

	/**
	 * \brief Constructs a matrix.
	 * If the number of rows, cols and batches are fixed on compile-time, they 
	 * must coincide with the sizes passed as arguments
	 * \param rows the number of rows
	 * \param cols the number of cols
	 * \param batches the number of batches
	 */
	Matrix(Index rows, Index cols, Index batches)
		: data_(rows, cols, batches)
	{}

    /**
    * \brief Constructs a matrix with the given data.
    * If the number of rows, cols and batches are fixed on compile-time, they
    * must coincide with the sizes passed as arguments
    * \param rows the number of rows
    * \param cols the number of cols
    * \param batches the number of batches
    */
    Matrix(const DevicePointer<_Scalar>& ptr, Index rows, Index cols, Index batches)
        : data_(ptr, rows, cols, batches)
    {}

	/**
	 * \brief Returns the number of rows of this matrix.
	 * \return the number of rows
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return data_.rows(); }

	/**
	 * \brief Returns the number of columns of this matrix.
	 * \return the number of columns
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return data_.cols(); }

	/**
	 * \brief Returns the number of batches of this matrix.
	 * \return the number of batches
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return data_.batches(); }

	// COEFFICIENT ACCESS

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
	 * \brief Computes the linear index from the three coordinates row, column and batch
	 * \param row the row index
	 * \param col the column index
	 * \param batch the batch index
	 * \return the linear index
	 * \see setRawCoeff(Index, Scalar)
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index index(Index row, Index col, Index batch) const
	{
		CUMAT_ASSERT_CUDA(row >= 0);
		CUMAT_ASSERT_CUDA(row < rows());
		CUMAT_ASSERT_CUDA(col >= 0);
		CUMAT_ASSERT_CUDA(col < cols());
		CUMAT_ASSERT_CUDA(batch >= 0);
		CUMAT_ASSERT_CUDA(batch < batches());
		if (CUMAT_IS_ROW_MAJOR(Flags)) {
			return col + cols() * (row + rows() * batch);
		}
		else {
			return row + rows() * (col + cols() * batch);
		}
	}

    //TODO: optimized path: implement a method 'sameLayout' that uses the linear index
    //directly instead of the row+col+batch.

	/**
	 * \brief Accesses the coefficient at the specified coordinate for reading and writing.
	 * If the device supports it (CUMAT_ASSERT_CUDA is defined), the
	 * access is checked for out-of-bound tests by assertions.
	 * \param row the row index
	 * \param col the column index
	 * \param batch the batch index
	 * \return a reference to the entry
	 */
	__device__ CUMAT_STRONG_INLINE _Scalar& coeff(Index row, Index col, Index batch, Index /*index*/)
	{
		return data_.data()[index(row, col, batch)];
	}
	/**
	* \brief Accesses the coefficient at the specified coordinate for reading.
	* If the device supports it (CUMAT_ASSERT_CUDA is defined), the
	* access is checked for out-of-bound tests by assertions.
	* Requirement of \c AccessFlags::ReadCwise 
	* \param row the row index
	* \param col the column index
	* \param batch the batch index
	* \return a read-only reference to the entry
	*/
	__device__ CUMAT_STRONG_INLINE _Scalar coeff(Index row, Index col, Index batch, Index /*index*/) const
	{
		Index idx = index(row, col, batch);
		//printf("[Thread %06d] memread %p at %i\n", int(blockIdx.x * blockDim.x + threadIdx.x), data_.data(), int(idx));
		return cuda::load(data_.data() + idx);
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
		CUMAT_ASSERT_CUDA(index >= 0);
		CUMAT_ASSERT_CUDA(index < size());
		data_.data()[index] = newValue;
	}

	/**
	* \brief Access to the linearized coefficient, read-only.
	* The format of the indexing depends on whether this
	* matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
	* Requirement of \c AccessFlags::RWCwise .
	* \param index the linearized index of the entry.
	* \return the entry at that index
	*/
	__device__ CUMAT_STRONG_INLINE _Scalar getRawCoeff(Index index) const
	{
		CUMAT_ASSERT_CUDA(index >= 0);
		CUMAT_ASSERT_CUDA(index < size());
		//printf("[Thread %06d] memread %p at %i\n", int(blockIdx.x * blockDim.x + threadIdx.x), data_.data(), int(index));
		return cuda::load(data_.data() + index);
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
        CUMAT_ASSERT_CUDA(index >= 0);
        CUMAT_ASSERT_CUDA(index < size());
        return data_.data()[index];
    }

	/**
	 * \brief Allows raw read and write access to the underlying buffer.
	 * Requirement of \c AccessFlags::WriteDirect.
	 * \return the underlying device buffer
	 */
	__host__ __device__ CUMAT_STRONG_INLINE _Scalar* data()
	{
		return data_.data();
	}

	/**
	* \brief Allows raw read-only access to the underlying buffer.
	* Requirement of \c AccessFlags::ReadDirect. 
	* \return the underlying device buffer
	*/
	__host__ __device__ CUMAT_STRONG_INLINE const _Scalar* data() const
	{
		return data_.data();
	}

	CUMAT_STRONG_INLINE const DevicePointer<_Scalar>& dataPointer() const
	{
		return data_.dataPointer();
	}

	CUMAT_STRONG_INLINE DevicePointer<_Scalar>& dataPointer()
	{
		return data_.dataPointer();
	}

    /**
     * \brief Checks if the this matrix has exclusive use to the underlying data, i.e. no other matrix expression shares the data.
     * This allows to check if this matrix is used in other expressions because that increments the internal reference counter.
     * If the internal reference counter is one, the matrix is nowhere else copied and this method returns true.
     * 
     * This is used to determine if the matrix can be modified inplace.
     * \return 
     */
    CUMAT_STRONG_INLINE bool isExclusiveUse() const
    {
        return data_.dataPointer().getCounter() == 1;
    }

    /**
    * \brief Checks if the underlying data is used by an other matrix expression and if so,
    * copies the data so that this matrix is the exclusive user of that data.
    *
    * This method has no effect if \ref isExclusiveUse is already true.
    *
    * Postcondition: <code>isExclusiveUse() == true</code>
    */
    void makeExclusiveUse()
    {
        if (isExclusiveUse()) return;

        DevicePointer<_Scalar> ptr = data_.dataPointer();
        data_ = Storage_t(rows(), cols(), batches());
        CUMAT_SAFE_CALL(cudaMemcpyAsync(data(), ptr.pointer(), sizeof(_Scalar)*rows()*cols()*batches(), cudaMemcpyDeviceToDevice, Context::current().stream()));
        CUMAT_PROFILING_INC(MemcpyDeviceToDevice);

        assert(isExclusiveUse());
    }

	// COPY CALLS

    /**
     * \brief Initializes a matrix from the given fixed-size 3d array.
     * This is intended to be used for small tests.
     * 
     * Example:
     \code
     int data[2][4][3] = {
        {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10,11,12}
        },
        {
            {13,14,15},
            {16,17,18},
            {19,20,21},
            {22,23,24}
        }
    };
    cuMat::BMatrixXiR m = cuMat::BMatrixXiR::fromArray(data);
    REQUIRE(m.rows() == 4);
    REQUIRE(m.cols() == 3);
    REQUIRE(m.batches() == 2);
    \endcode
     * 
     * Note that the returned matrix is always of row-major storage.
     * (This is how arrays are stored in C++)
     * 
     * \tparam Rows the number of rows, infered from the passed argument
     * \tparam Cols the number of columns, infered from the passed argument
     * \tparam Batches the number of batches, infered from the passed argument
     * \param a the fixed-side 3d array used to initialize the matrix
     * \return A row-major matrix with the specified contents
     */
    template<int Rows, int Cols, int Batches>
    static Matrix<_Scalar, ((Rows>1) ? Dynamic : 1), ((Cols>1) ? Dynamic : 1), ((Batches>1) ? Dynamic : 1), RowMajor>
        fromArray(const _Scalar (&a)[Batches][Rows][Cols])
    {
        typedef Matrix<_Scalar, (Rows > 1) ? Dynamic : Rows, (Cols > 1) ? Dynamic : Cols, (Batches > 1) ? Dynamic : Batches, RowMajor> mt;
        mt m(Rows, Cols, Batches);
        m.copyFromHost((const _Scalar*)a);
        CUMAT_PROFILING_INC(MemcpyHostToDevice);
        return m;
    }

	/**
	 * \brief Performs a synchronous copy from host data into the
	 * device memory of this matrix.
	 * This copy is synchronized on the default stream,
	 * hence synchronous to every computation but slow.
	 * \param data the data to copy into this matrix
	 */
	void copyFromHost(const _Scalar* data)
	{
		//slower, conservative: full synchronization
		//CUMAT_SAFE_CALL(cudaMemcpy(data_.data(), data, sizeof(_Scalar)*size(), cudaMemcpyHostToDevice));
		//CUMAT_SAFE_CALL(cudaDeviceSynchronize());

		//faster: only synchronize this stream
		CUMAT_SAFE_CALL(cudaMemcpyAsync(data_.data(), data, sizeof(_Scalar)*size(), cudaMemcpyHostToDevice, Context::current().stream()));
		CUMAT_SAFE_CALL(cudaStreamSynchronize(Context::current().stream()));

        CUMAT_PROFILING_INC(MemcpyHostToDevice);
	}

	/**
	* \brief Performs a synchronous copy from the
	* device memory of this matrix into the
	* specified host memory
	* This copy is synchronized on the default stream,
	 * hence synchronous to every computation but slow.
	* \param data the data in which the matrix is stored
	*/
	void copyToHost(_Scalar* data) const
	{
	    //slower, conservative: full synchronization
		//CUMAT_SAFE_CALL(cudaStreamSynchronize(Context::current().stream()));
		//CUMAT_SAFE_CALL(cudaMemcpy(data, data_.data(), sizeof(_Scalar)*size(), cudaMemcpyDeviceToHost));

		//faster: only synchronize this stream
		CUMAT_SAFE_CALL(cudaMemcpyAsync(data, data_.data(), sizeof(_Scalar)*size(), cudaMemcpyDeviceToHost, Context::current().stream()));
		CUMAT_SAFE_CALL(cudaStreamSynchronize(Context::current().stream()));

        CUMAT_PROFILING_INC(MemcpyDeviceToHost);
	}

	// EIGEN INTEROP
#if CUMAT_EIGEN_SUPPORT==1

	/**
	 * \brief The Eigen Matrix type that corresponds to this cuMat matrix.
	 * Note that Eigen does not support batched matrices. Hence, you can
	 * only convert cuMat matrices of batch size 1 (during compile time or runtime)
	 * to Eigen.
	 */
	typedef typename CUMAT_NAMESPACE eigen::MatrixCuMatToEigen<Type>::type EigenMatrix_t;

	/**
	 * \brief Converts this cuMat matrix to the corresponding Eigen matrix.
	 * Note that Eigen does not support batched matrices. Hence, this 
	 * conversion is only possible, if<br>
	 * a) the matrix has a compile-time batch size of 1, or<br>
	 * b) the matrix has a dynamic batch size and the batch size is 1 during runtime.
	 * 
	 * <p>
	 * Design decision:<br>
	 * Converting between cuMat and Eigen is done using synchronous memory copies.
	 * It requires a complete synchronization of host and device. Therefore,
	 * this operation is very expensive.<br>
	 * Because of that, I decided to implement the conversion using 
	 * explicit methods (toEigen() and fromEigen(EigenMatrix_t) 
	 * instead of conversion operators or constructors.
	 * It should be made clear to the reader that this operation
	 * is expensive and should be used carfully, i.e. only to pass
	 * data in and out before and after the computation.
	 * \return the Eigen matrix with the contents of this matrix.
	 */
	EigenMatrix_t toEigen() const
	{
		CUMAT_STATIC_ASSERT(_Batches == 1 || _Batches == Dynamic, "Compile-time batches>1 not allowed. Eigen does not support batches");
		if (_Batches == Dynamic) CUMAT_ASSERT_ARGUMENT(batches() == 1);
		EigenMatrix_t mat(rows(), cols());
		copyToHost(mat.data());
		return mat;
	}

	/**
	* \brief Converts the specified Eigen matrix into the
	* corresponding cuMat matrix.
	* Note that Eigen does not support batched matrices. Hence, this
	* conversion is only possible, if<br>
	* a) the target matrix has a compile-time batch size of 1, or<br>
	* b) the target matrix has a dynamic batch size and the batch size is 1 during runtime.<br>
	* A new cuMat matrix is returned.
	*
	* <p>
	* Design decision:<br>
	* Converting between cuMat and Eigen is done using synchronous memory copies.
	* It requires a complete synchronization of host and device. Therefore,
	* this operation is very expensive.<br>
	* Because of that, I decided to implement the conversion using
	* explicit methods (toEigen() and fromEigen(EigenMatrix_t)
	* instead of conversion operators or constructors.
	* It should be made clear to the reader that this operation
	* is expensive and should be used carfully, i.e. only to pass
	* data in and out before and after the computation.
	* \return the Eigen matrix with the contents of this matrix.
	*/
	static Type fromEigen(const EigenMatrix_t& mat)
	{
		CUMAT_STATIC_ASSERT(_Batches == 1 || _Batches == Dynamic, "Compile-time batches>1 not allowed. Eigen does not support batches");
		Type m(mat.rows(), mat.cols());
		m.copyFromHost(reinterpret_cast<const _Scalar*>(mat.data()));
		return m;
	}

#endif

	// ASSIGNMENT

	//template<typename OtherDerieved>
	//CUMAT_STRONG_INLINE Derived& operator=(const MatrixBase<OtherDerieved>& other);

	//assignments from other matrices: convert compile-size to dynamic

    /**
	 * \brief Shallow copy constructor, the underlying data is shared!!
	 * This only works if the dimension other matrix is compatible with the
	 * static dimension of this matrix.
	 * \tparam _OtherRows 
	 * \tparam _OtherColumns 
	 * \tparam _OtherBatches 
	 * \tparam _OtherFlags 
	 * \param other 
	 */
	template<int _OtherRows, int _OtherColumns, int _OtherBatches, int _OtherFlags>
	__host__ Matrix(const Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, _OtherFlags>& other)
		: data_(other.dataPointer(), other.rows(), other.cols(), other.batches()) //shallow copy
	{
		CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Rows != Dynamic && _OtherRows != Dynamic, _OtherRows == _Rows), 
			"unable to assign a matrix to another matrix with a different compile time row count");
		CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Columns != Dynamic && _OtherColumns != Dynamic, _OtherColumns == _Columns),
			"unable to assign a matrix to another matrix with a different compile time column count");
		CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Batches != Dynamic && _OtherBatches != Dynamic, _OtherBatches == _Batches),
			"unable to assign a matrix to another matrix with a different compile time batch count");

        CUMAT_ASSERT_DIMENSION(CUMAT_IMPLIES(_Rows != Dynamic, _Rows == other.rows()));
        CUMAT_ASSERT_DIMENSION(CUMAT_IMPLIES(_Columns != Dynamic, _Columns == other.cols()));
        CUMAT_ASSERT_DIMENSION(CUMAT_IMPLIES(_Batches != Dynamic, _Batches == other.batches()));

		//Only allow implicit transposing if we have vectors
		CUMAT_STATIC_ASSERT(_OtherFlags == _Flags || _Rows==1 || _Columns==1,
			"unable to assign a matrix to another matrix with a different storage order, transpose them explicitly");
	}

    /**
	 * \brief Shallow assignment operator, the underlying data is shared!
	 * The data of this matrix is replaced by the other matrix.
	 * This only works if the dimension other matrix is compatible with the
	 * static dimension of this matrix.
	 * \tparam _OtherRows 
	 * \tparam _OtherColumns 
	 * \tparam _OtherBatches 
	 * \tparam _OtherFlags 
	 * \param other the other matrix
	 * \return this
	 */
	template<int _OtherRows, int _OtherColumns, int _OtherBatches, int _OtherFlags>
	CUMAT_STRONG_INLINE Type& operator=(const Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, _OtherFlags>& other)
	{
		CUMAT_STATIC_ASSERT(_Rows == Dynamic || _OtherRows == _Rows,
			"unable to assign a matrix to another matrix with a different compile time row count");
		CUMAT_STATIC_ASSERT(_Columns == Dynamic || _OtherColumns == _Columns,
			"unable to assign a matrix to another matrix with a different compile time column count");
		CUMAT_STATIC_ASSERT(_Batches == Dynamic || _OtherBatches == _Batches,
			"unable to assign a matrix to another matrix with a different compile time batch count");

		//Only allow implicit transposing if we have vectors
		CUMAT_STATIC_ASSERT(_OtherFlags == _Flags || _Rows==1 || _Columns==1,
			"unable to assign a matrix to another matrix with a different storage order, transpose them explicitly");

		// shallow copy
		data_ = Storage_t(other.dataPointer(), other.rows(), other.cols(), other.batches());

		return *this;
	}

	// EVALUATIONS

    /**
	 * \brief Evaluation constructor, new memory is allocated for the result.
	 * \tparam Derived 
	 * \param expr the matrix expression
	 */
	template<typename Derived>
	Matrix(const MatrixBase<Derived>& expr)
		: data_(expr.rows(), expr.cols(), expr.batches())
	{
		//expr.template evalTo<Type, AssignmentMode::ASSIGN>(*this);
        internal::Assignment<Type, Derived, AssignmentMode::ASSIGN, internal::DenseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());
	}

    /**
	 * \brief Evaluation assignment, new memory is allocated for the result.
	 * Exception: if it can be guarantered, that the memory is used exclusivly (\ref isExclusiveUse() returns true),
	 * and the size of this matrix and the result match, the memory is reuse.
	 * \tparam Derived 
	 * \param expr 
	 * \return 
	 */
	template<typename Derived>
	CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
	{
        if (!isExclusiveUse() || rows() != expr.rows() || cols() != expr.cols() || batches() != expr.batches()) {
            //allocate new memory for the result
            data_ = Storage_t(expr.rows(), expr.cols(), expr.batches());
        } //else: reuse memory
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
        CUMAT_ERROR_IF_NO_NVCC(compoundAssignment)                                                      \
        internal::Assignment<Type, Derived, AssignmentMode:: mode , internal::DenseDstTag, typename internal::traits<Derived>::SrcTag>::assign(*this, expr.derived());   \
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
		CUMAT_ERROR_IF_NO_NVCC(inplaceMultiplication)
        using Expr = decltype(Type::Constant(rows(), cols(), batches(), scalar));
        internal::Assignment<Type, Expr, AssignmentMode::MUL, internal::DenseDstTag, typename internal::traits<Expr>::SrcTag>::assign(*this, Type::Constant(rows(), cols(), batches(), scalar));
        return *this;
	}

    /**
    * \brief Explicit overloading of \c operator*= for matrix right hand sides.
    * It performs an inplace matrix multiplication \code *this = *this * rhs \endcode.
    * Note that \c rhs must be square because the size of *this must not change.
    * 
    * This is needed to disambiguate the difference between component-wise operations and matrix operations.
    * All other compount-assignment operators (+=, -=, /=, ...) act component-wise.
    *
    * \tparam _Derived the type of the matrix
    * \param rhs the right hand side matrix
    * \return *this
    */
    template<typename _Derived>
    CUMAT_STRONG_INLINE Type& operator*= (const MatrixBase<_Derived>& rhs)
    {
		CUMAT_ERROR_IF_NO_NVCC(inplaceMatmul)
        //check that the rhs is in fact square
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(internal::traits<_Derived>::RowsAtCompileTime != Dynamic && internal::traits<_Derived>::ColsAtCompileTime != Dynamic,
            internal::traits<_Derived>::RowsAtCompileTime == internal::traits<_Derived>::ColsAtCompileTime),
            "The right hand side must be a static matrix");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Columns != Dynamic && internal::traits<_Derived>::RowsAtCompileTime != Dynamic,
            _Columns == internal::traits<_Derived>::RowsAtCompileTime),
            "The right hand side is not compatible with this matrix");
        CUMAT_ASSERT_DIMENSION(rhs.rows() == rhs.cols());
        CUMAT_ASSERT_DIMENSION(cols() == rhs.rows());
        //check that the batch size matches (broadcasting only over rhs)
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Batches != Dynamic && internal::traits<_Derived>::BatchesAtCompileTime != Dynamic,
            _Batches == internal::traits<_Derived>::BatchesAtCompileTime || internal::traits<_Derived>::BatchesAtCompileTime==1),
            "Batches must match or attempt to broadcast over this matrix");
        CUMAT_ASSERT_DIMENSION(batches() == rhs.batches() || rhs.batches() == 1);

        //cuBLAS GEMM can't work inplace -> clone this and place the result inplace
        inplace() = deepClone() * rhs;

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
    internal::MatrixInplaceAssignment<Type> inplace()
	{
        return internal::MatrixInplaceAssignment<Type>(this);
	}

private:

    template<int _OtherRows, int _OtherColumns, int _OtherBatches>
    void deepCloneImpl(Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, Flags>& mat) const
    {
        //optimized path: direct memcpy
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Rows!=Dynamic && _OtherRows!=Dynamic, _Rows == _OtherRows), 
            "unable to assign a matrix to another matrix with a different compile time row count");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Columns!=Dynamic && _OtherColumns!=Dynamic, _Columns == _OtherColumns), 
            "unable to assign a matrix to another matrix with a different compile time column count");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Batches!=Dynamic && _OtherBatches!=Dynamic, _Batches == _OtherBatches), 
            "unable to assign a matrix to another matrix with a different compile time row count");
            
        CUMAT_ASSERT(rows() == mat.rows());
        CUMAT_ASSERT(cols() == mat.cols());
        CUMAT_ASSERT(batches() == mat.batches());
        
        CUMAT_SAFE_CALL(cudaMemcpyAsync(mat.data(), data(), sizeof(_Scalar)*rows()*cols()*batches(), cudaMemcpyDeviceToDevice, Context::current().stream()));
        CUMAT_PROFILING_INC(MemcpyDeviceToDevice);
    }
    
    template<int _OtherRows, int _OtherColumns, int _OtherBatches>
    void deepCloneImpl_directTranspose(Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, TransposedFlags>& mat, std::integral_constant<bool, true>) const
    {
        //optimized path: direct transpose
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Rows!=Dynamic && _OtherRows!=Dynamic, _Rows == _OtherRows), 
            "unable to assign a matrix to another matrix with a different compile time row count");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Columns!=Dynamic && _OtherColumns!=Dynamic, _Columns == _OtherColumns), 
            "unable to assign a matrix to another matrix with a different compile time column count");
        CUMAT_STATIC_ASSERT(CUMAT_IMPLIES(_Batches!=Dynamic && _OtherBatches!=Dynamic, _Batches == _OtherBatches), 
            "unable to assign a matrix to another matrix with a different compile time row count");
            
        CUMAT_ASSERT(rows() == mat.rows());
        CUMAT_ASSERT(cols() == mat.cols());
        CUMAT_ASSERT(batches() == mat.batches());
        
        Index m = CUMAT_IS_ROW_MAJOR(Flags) ? rows() : cols();
        Index n = CUMAT_IS_ROW_MAJOR(Flags) ? cols() : rows();
        internal::directTranspose<_Scalar>(mat.data(), data(), m, n, batches());
    }
    template<int _OtherRows, int _OtherColumns, int _OtherBatches>
    void deepCloneImpl_directTranspose(Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, TransposedFlags>& mat, std::integral_constant<bool, false>) const
    {
        //cuBLAS is not available for that type
        //default: cwise evaluation
		typedef Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, TransposedFlags> TargetType;
		internal::Assignment<TargetType, Type, AssignmentMode::ASSIGN, internal::DenseDstTag, internal::CwiseSrcTag>::assign(mat, *this);
    }

    template<int _OtherRows, int _OtherColumns, int _OtherBatches>
    void deepCloneImpl(Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, TransposedFlags>& mat) const
    {
        deepCloneImpl_directTranspose(mat, std::integral_constant<bool, internal::NumTraits<_Scalar>::IsCudaNumeric>());
    }

    //template<typename Derived>
    //void deepCloneImpl(MatrixBase<Derived>& m) const
    //{
    //    //default: cwise evaluation
    //    internal::Assignment<Derived, Type, AssignmentMode::ASSIGN, internal::DenseDstTag, internal::CwiseSrcTag>::assign(m.derived(), *this);
    //}

public:

    /**
     * \brief Performs a deep clone of the matrix.
     * Usually, when you assign matrix instances, the underlying data is shared. This method explicitly copies the internal data
     * of the matrix into a new matrix. The two matrices are then completely independent, i.e. if you perform an 
     * inplace modification on this or the returned matrix, the changes are not reflected in the other.
     * 
     * Furthermore, this is the only option to explicitly change the storage mode:
     * In cuMat, implicit transposition is not allowed when assigning matrices of different storage mode (column major vs. row major) to each other.
     * This method, however, allows to specify the target storage mode.
     * By default, this is the same as the current mode, resulting in a simple memcpy
     *
     * \tparam _TargetFlags the target storage mode (ColumnMajor or RowMajor). Default: the current storage mode
     * \return the new matrix with a deep clone of the data
     */
    template<int _TargetFlags = _Flags>
    CUMAT_STRONG_INLINE Matrix<_Scalar, _Rows, _Columns, _Batches, _TargetFlags> deepClone() const
	{
        Matrix<_Scalar, _Rows, _Columns, _Batches, _TargetFlags> mat(rows(), cols(), batches());
        deepCloneImpl(mat);
        return mat;
	}

	// STATIC METHODS AND OTHER HELPERS
    /**
	 * \brief Sets all entries to zero.
	 * Warning: this operation works in-place and therefore violates the copy-on-write paradigm.
	 */
	void setZero()
	{
		Index s = size();
		if (s > 0) {
			CUMAT_SAFE_CALL(cudaMemsetAsync(data(), 0, sizeof(_Scalar) * size(), Context::current().stream()));
		}
	}

#include "MatrixNullaryOpsPlugin.inl"
#include "MatrixBlockPluginLvalue.inl"

    //TODO: find a better place for the following two methods:

#ifdef CUMAT_PARSED_BY_DOXYGEN
    /**
    * \brief Extracts the real part of the complex matrix.
    * This method is only available for complex matrices.
    * This is the non-const lvalue version
    */
    ExtractComplexPartOp<Type, false, true> real()
    {
        CUMAT_STATIC_ASSERT(internal::NumTraits<_Scalar>::IsComplex, "Matrix must be complex");
        return ExtractComplexPartOp<Type, false, true>(*this);
    }
#else
    /**
    * \brief Extracts the real part of the complex matrix.
    * Specialization for non-complex matrices: no-op.
    * This is the non-const lvalue version.
    */
    template<typename S = typename internal::traits<Type>::Scalar, 
            typename = typename std::enable_if<!internal::NumTraits<S>::IsComplex>::type>
    const Type& real()
    {
        return *this;
    }
    /**
    * \brief Extracts the real part of the complex matrix.
    * Specialization for complex matrices, extracts the real part.
    * This is the non-const lvalue version
    */
    template<typename S = typename internal::traits<Type>::Scalar, 
            typename = typename std::enable_if<internal::NumTraits<S>::IsComplex>::type>
    ExtractComplexPartOp<Type, false, true> real()
    {
        CUMAT_STATIC_ASSERT(internal::NumTraits<_Scalar>::IsComplex, "Matrix must be complex");
        return ExtractComplexPartOp<Type, false, true>(*this);
    }
#endif
    /**
    * \brief Extracts the imaginary part of the complex matrix.
    * This method is only available for complex matrices.
    * This is the non-const lvalue version
    */
    ExtractComplexPartOp<Type, true, true> imag()
    {
        CUMAT_STATIC_ASSERT(internal::NumTraits<_Scalar>::IsComplex, "Matrix must be complex");
        return ExtractComplexPartOp<Type, true, true>(*this);
    }

};

/**
 * \brief Custom operator<< that prints the matrix and additional information.
 * First, information about the matrix like shape and storage options are printed,
 * followed by the whole matrix batch-by-batch.
 * 
 * This operations involves copying the matrix from device to host.
 * It is slow, use it only for debugging purpose.
 * \param os the output stream
 * \param m the matrix
 * \return the output stream again
 */
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
__host__ std::ostream& operator<<(std::ostream& os, const Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags>& m)
{
    os << "Matrix: ";
    os << "rows=" << m.rows() << " (" << (_Rows == Dynamic ? "dynamic" : "compile-time") << ")";
    os << ", cols=" << m.cols() << " (" << (_Columns == Dynamic ? "dynamic" : "compile-time") << ")";
    os << ", batches=" << m.batches() << " (" << (_Batches == Dynamic ? "dynamic" : "compile-time") << ")";
    os << ", storage=" << (CUMAT_IS_ROW_MAJOR(_Flags) ? "Row-Major" : "Column-Major") << std::endl;

	//New code, custom printing implementation
	io::print_matrix(os, m, io::IOFormat());
	
	//Old code for reference, uses Eigen
    //for (int batch = 0; batch < m.batches(); ++batch)
    //{
    //    const auto emat = m.template block<Dynamic, Dynamic, 1>(0, 0, batch, m.rows(), m.cols(), 1).eval().toEigen();
    //    if (m.batches() > 1) os << "batch " << batch << std::endl;
    //    os << emat << std::endl;
    //}

	return os;
}

namespace internal
{
    template<
        typename _Dst, 
        typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags, 
        AssignmentMode _Mode>
    struct Assignment<_Dst, Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags>, _Mode, DenseDstTag, CwiseSrcTag>
    {
        typedef Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> Type;
        static void assign(_Dst& dst, const Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags>& src)
        {
            //Here is now the place to perform the fast track evaluations for specific destinations
            //Memcopy + Direct transpose, if dst is a Matrix or a MatrixBlock of specific shape (a slice)
            //For now, just delegate to Cwise evaluation
            Assignment<_Dst, CwiseOp<Type>, _Mode, DenseDstTag, CwiseSrcTag>::assign(dst, src);
        }
    };

    template<typename _Matrix>
    class MatrixInplaceAssignment
    {
    private:
        _Matrix * matrix_;
    public:
        MatrixInplaceAssignment(_Matrix* matrix) : matrix_(matrix) {}

        /**
         * \brief Evaluates the expression inline inplace into the current matrix.
         * No new memory is created, it is reused!
         * This operator fails with an exception if the dimensions don't match.
         * \param expr the other expression
         * \return the underlying matrix
         */
        template<typename Derived>
        CUMAT_STRONG_INLINE _Matrix& operator=(const MatrixBase<Derived>& expr)
        {
            CUMAT_ASSERT_DIMENSION(matrix_->rows() == expr.rows());
            CUMAT_ASSERT_DIMENSION(matrix_->cols() == expr.cols());
            CUMAT_ASSERT_DIMENSION(matrix_->batches() == expr.batches());
            Assignment<_Matrix, Derived, AssignmentMode::ASSIGN, DenseDstTag, typename Derived::SrcTag>::assign(*matrix_, expr.derived());
            return *matrix_;
        }
    };
}

//Common typedefs

/** \defgroup matrixtypedefs Global matrix typedefs
*
* cuMat defines several typedef shortcuts for most common matrix and vector types.
*
* The general patterns are the following:
*
* \c MatrixSizeType where \c Size can be \c 2,\c 3,\c 4 for fixed size square matrices or \c X for dynamic size,
* and where \c Type can be \c b for boolean, \c i for integer, \c f for float, \c d for double, \c cf for complex float, \c cd
* for complex double.
* Further, the suffix \c C indicates ColumnMajor storage, \c R RowMajor storage. The default (no suffix) is ColumnMajor.
* The prefix \c B indicates batched matrices of dynamic batch size. Typedefs without this prefix have a compile-time batch size of 1.
*
* For example, \c BMatrix3dC is a fixed-size 3x3 matrix type of doubles but with dynamic batch size,
*  and \c MatrixXf is a dynamic-size matrix of floats, non-batched.
*
* There are also \c VectorSizeType and \c RowVectorSizeType which are self-explanatory. For example, \c Vector4cf is
* a fixed-size vector of 4 complex floats.
*
* \sa class Matrix
*/

#define CUMAT_DEF_MATRIX1(scalar1, scalar2, order1, order2) \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 1, 1, order1> Scalar ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 1, Dynamic, order1> BScalar ## scalar2 ## order2; \
    \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 2, 1, 1, order1> Vector2 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 2, 1, Dynamic, order1> BVector2 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 3, 1, 1, order1> Vector3 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 3, 1, Dynamic, order1> BVector3 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 4, 1, 1, order1> Vector4 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 4, 1, Dynamic, order1> BVector4 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, Dynamic, 1, 1, order1> VectorX ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, Dynamic, 1, Dynamic, order1> BVectorX ## scalar2 ## order2; \
    \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 2, 1, order1> RowVector2 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 2, Dynamic, order1> BRowVector2 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 3, 1, order1> RowVector3 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 3, Dynamic, order1> BRowVector3 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 4, 1, order1> RowVector4 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, 4, Dynamic, order1> BRowVector4 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, Dynamic, 1, order1> RowVectorX ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 1, Dynamic, Dynamic, order1> BRowVectorX ## scalar2 ## order2; \
    \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 2, 2, 1, order1> Matrix2 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 2, 2, Dynamic, order1> BMatrix2 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 3, 3, 1, order1> Matrix3 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 3, 3, Dynamic, order1> BMatrix3 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 4, 4, 1, order1> Matrix4 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, 4, 4, Dynamic, order1> BMatrix4 ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, Dynamic, Dynamic, 1, order1> MatrixX ## scalar2 ## order2; \
    /** \ingroup matrixtypedefs */ typedef Matrix<scalar1, Dynamic, Dynamic, Dynamic, order1> BMatrixX ## scalar2 ## order2; \

#define CUMAT_DEF_MATRIX2(scalar1, scalar2) \
    CUMAT_DEF_MATRIX1(scalar1, scalar2, ColumnMajor, C) \
    CUMAT_DEF_MATRIX1(scalar1, scalar2, RowMajor, R) \
    CUMAT_DEF_MATRIX1(scalar1, scalar2, ColumnMajor, )

CUMAT_DEF_MATRIX2(bool, b)
CUMAT_DEF_MATRIX2(int, i)
CUMAT_DEF_MATRIX2(long, l)
CUMAT_DEF_MATRIX2(long long, ll)
CUMAT_DEF_MATRIX2(float, f)
CUMAT_DEF_MATRIX2(double, d)
CUMAT_DEF_MATRIX2(cfloat, cf)
CUMAT_DEF_MATRIX2(cdouble, cd)

#undef CUMAT_DEF_MATRIX2
#undef CUMAT_DEF_MATRIX1


CUMAT_NAMESPACE_END

#endif
