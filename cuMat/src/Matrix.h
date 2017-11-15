#ifndef __CUMAT_MATRIX_H__
#define __CUMAT_MATRIX_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "NumTraits.h"
#include "CudaUtils.h"
#include "Constants.h"
#include "Context.h"
#include "DevicePointer.h"
#include "MatrixBase.h"
#include "CwiseOp.h"
#include "NullaryOps.h"

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
		typedef typename _Scalar Scalar;
		enum
		{
			Flags = _Flags,
			RowsAtCompileTime = _Rows,
			ColsAtCompileTime = _Columns,
			BatchesAtCompileTime = _Batches
		};
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
class Matrix 
	: public CwiseOp<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> > //inheriting from CwiseOp allows a Matrix to be used as a leaf
	//: public MatrixBase<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> >
{
protected:
	using Storage_t = internal::DenseStorage<_Scalar, _Rows, _Columns, _Batches>;
	Storage_t data_;
public:
	enum
	{
		Flags = _Flags,
		Rows = _Rows,
		Columns = _Columns,
		Batches = _Batches
	};
	using Scalar = _Scalar;
	using Type = Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags>;

	typedef typename MatrixBase<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> > Base;
	using Base::eval_t;
	using Base::size;

	/**
	 * \brief Default constructor.
	 * For completely fixed-size matrices, this creates a matrix of that size.
	 * For (fully or partially) dynamic matrices, creates a matrix of size 0.
	 */
    __host__ __device__
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
	 * \brief Computes the linear index from the three coordinates row, column and batch
	 * \param row the row index
	 * \param col the column index
	 * \param batch the batch index
	 * \return the linear index
	 * \see rawCoeff(Index)
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

	/**
	 * \brief Accesses the coefficient at the specified coordinate for reading and writing.
	 * If the device supports it (CUMAT_ASSERT_CUDA is defined), the
	 * access is checked for out-of-bound tests by assertions.
	 * \param row the row index
	 * \param col the column index
	 * \param batch the batch index
	 * \return a reference to the entry
	 */
	__device__ CUMAT_STRONG_INLINE _Scalar& coeff(Index row, Index col, Index batch)
	{
		return data_.data()[index(row, col, batch)];
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
	__device__ CUMAT_STRONG_INLINE const _Scalar& coeff(Index row, Index col, Index batch) const
	{
		return cuda::load(data_.data() + index(row, col, batch));
	}

	/**
	 * \brief Access to the linearized coefficient.
	 * The format of the indexing depends on whether this
	 * matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
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
	* \brief Access to the linearized coefficient, read-only.
	* The format of the indexing depends on whether this
	* matrix is column major (ColumnMajorBit) or row major (RowMajorBit).
	* \param index the linearized index of the entry.
	* \return the entry at that index
	*/
	__device__ CUMAT_STRONG_INLINE const _Scalar& rawCoeff(Index index) const
	{
		CUMAT_ASSERT_CUDA(index >= 0);
		CUMAT_ASSERT_CUDA(index < size());
		return cuda::load(data_.data() + index);
	}

	/**
	 * \brief Allows raw read and write access to the underlying buffer
	 * \return the underlying device buffer
	 */
	__host__ __device__ CUMAT_STRONG_INLINE _Scalar* data()
	{
		return data_.data();
	}

	/**
	* \brief Allows raw read-only access to the underlying buffer
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
    static Matrix<_Scalar, (Rows>1) ? Dynamic : 1, (Cols>1) ? Dynamic : 1, (Batches>1) ? Dynamic : 1, RowMajor>
        fromArray(const _Scalar (&a)[Batches][Rows][Cols])
    {
        typedef Matrix<_Scalar, (Rows > 1) ? Dynamic : Rows, (Cols > 1) ? Dynamic : Cols, (Batches > 1) ? Dynamic : Batches, RowMajor> mt;
        mt m(Rows, Cols, Batches);
        m.copyFromHost((const _Scalar*)a);
        return m;
    }

	/**
	 * \brief Performs a sychronous copy from host data into the
	 * device memory of this matrix.
	 * This copy is synchronized on the default stream,
	 * hence synchronous to every computation but slow.
	 * \param data the data to copy into this matrix
	 */
	void copyFromHost(const _Scalar* data)
	{
		CUMAT_SAFE_CALL(cudaMemcpy(data_.data(), data, sizeof(_Scalar)*size(), cudaMemcpyHostToDevice));
	}

	/**
	* \brief Performs a sychronous copy from the
	* device memory of this matrix into the
	* specified host memory
	* This copy is synchronized on the default stream,
	 * hence synchronous to every computation but slow.
	* \param data the data in which the matrix is stored
	*/
	void copyToHost(_Scalar* data) const
	{
		CUMAT_SAFE_CALL(cudaMemcpy(data, data_.data(), sizeof(_Scalar)*size(), cudaMemcpyDeviceToHost));
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

#ifdef CUMAT_PARSED_BY_DOXYGEN
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
	EigenMatrix_t toEigen() const;

	/**
	* \brief Converts the specified Eigen matrix into the
	* corresponding cuMat matrix.
	* Note that Eigen does not support batched matrices. Hence, this
	* conversion is only possible, if<br>
	* a) the target matrix has a compile-time batch size of 1, or<br>
	* b) the target matrix has a dynamic batch size and the batch size is 1 during runtime.<br>
	* A new cuMat matrix is returned.
	* TODO: implement this as an expression template
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
	static Type fromEigen(const EigenMatrix_t& mat);
#else

	//TODO: once expression templates are implemented,
	// move them upward to work on the expression templates directly
	// (issue an evaluation in between)
	template<typename T = std::enable_if<(_Batches == 1 || _Batches == Dynamic), EigenMatrix_t>>
	typename T::type toEigen() const
	{
		if (_Batches == Dynamic) CUMAT_ASSERT_ARGUMENT(batches() == 1);
		EigenMatrix_t mat(rows(), cols());
		copyToHost(mat.data());
		return mat;
	}

	//TODO: once expression templates are implemented,
	// return an expression template that can then be assigned
	// to a matrix. By that, the matrix can also be reshaped.
	template<typename T = std::enable_if<_Batches == 1 || _Batches == Dynamic, Type>>
	static typename T::type fromEigen(const EigenMatrix_t& mat)
	{
		Type m(mat.rows(), mat.cols());
		m.copyFromHost(reinterpret_cast<const _Scalar*>(mat.data()));
		return m;
	}
#endif
#endif

	// ASSIGNMENT

	//template<typename OtherDerieved>
	//CUMAT_STRONG_INLINE Derived& operator=(const MatrixBase<OtherDerieved>& other);

	//assignments from other matrices: convert compile-size to dynamic

	template<int _OtherRows, int _OtherColumns, int _OtherBatches, int _OtherFlags>
	Matrix(const Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, _OtherFlags>& other)
		: data_(other.dataPointer(), other.rows(), other.cols(), other.batches()) //shallow copy
	{
		CUMAT_STATIC_ASSERT(_Rows == Dynamic || _OtherRows == _Rows, 
			"unable to assign a matrix to another matrix with a different compile time row count");
		CUMAT_STATIC_ASSERT(_Columns == Dynamic || _OtherColumns == _Columns, 
			"unable to assign a matrix to another matrix with a different compile time column count");
		CUMAT_STATIC_ASSERT(_Batches == Dynamic || _OtherBatches == _Batches, 
			"unable to assign a matrix to another matrix with a different compile time batch count");

		//TODO: relax the following constraint to allow automatic transposing?
		CUMAT_STATIC_ASSERT(_OtherFlags == _Flags,
			"unable to assign a matrix to another matrix with a different storage order, transpose them explicitly");
	}

	template<int _OtherRows, int _OtherColumns, int _OtherBatches, int _OtherFlags>
	CUMAT_STRONG_INLINE Type& operator=(const Matrix<_Scalar, _OtherRows, _OtherColumns, _OtherBatches, _OtherFlags>& other)
	{
		CUMAT_STATIC_ASSERT(_Rows == Dynamic || _OtherRows == _Rows,
			"unable to assign a matrix to another matrix with a different compile time row count");
		CUMAT_STATIC_ASSERT(_Columns == Dynamic || _OtherColumns == _Columns,
			"unable to assign a matrix to another matrix with a different compile time column count");
		CUMAT_STATIC_ASSERT(_Batches == Dynamic || _OtherBatches == _Batches,
			"unable to assign a matrix to another matrix with a different compile time batch count");

		//TODO: relax the following constraint to allow automatic transposing?
		CUMAT_STATIC_ASSERT(_OtherFlags == _Flags,
			"unable to assign a matrix to another matrix with a different storage order, transpose them explicitly");

		// shallow copy
		data_ = Storage_t(other.dataPointer(), other.rows(), other.cols(), other.batches());

		return *this;
	}

	// EVALUATIONS

	template<typename Derived>
	Matrix(const MatrixBase<Derived>& expr)
		: data_(expr.rows(), expr.cols(), expr.batches())
	{
		expr.evalTo(*this);
	}

	template<typename Derived>
	CUMAT_STRONG_INLINE Type& operator=(const MatrixBase<Derived>& expr)
	{
		data_ = Storage_t(expr.rows(), expr.cols(), expr.batches());
		expr.evalTo(*this);
		return *this;
	}

	// STATIC METHODS AND OTHER HELPERS

	template<typename _NullaryFunctor>
	using NullaryOp_t = NullaryOp<_Scalar, _Rows, _Columns, _Batches, _Flags, _NullaryFunctor >;

	static NullaryOp_t<functor::ConstantFunctor<_Scalar> >
	Constant(Index rows, Index cols, Index batches, const _Scalar& value)
	{
		if (_Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(_Rows == rows && "runtime row count does not match compile time row count");
		if (_Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(_Columns == cols && "runtime row count does not match compile time row count");
		if (_Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(_Batches == batches && "runtime row count does not match compile time row count");
		return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
			rows, cols, batches, functor::ConstantFunctor<_Scalar>(value));
	}
	//Specialization for some often used cases
	template<typename T = std::enable_if<_Batches!=Dynamic && _Rows==Dynamic && _Columns==Dynamic, 
		NullaryOp_t<functor::ConstantFunctor<_Scalar> >>>
	static T::type Constant(Index rows, Index cols, const _Scalar& value)
	{
		return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
			rows, cols, _Batches, functor::ConstantFunctor<_Scalar>(value));
	}
	template<typename T = std::enable_if<_Batches != Dynamic 
		&& ((_Rows == Dynamic && _Columns != Dynamic) || (_Rows != Dynamic && _Columns == Dynamic)),
		NullaryOp_t<functor::ConstantFunctor<_Scalar> >>>
		static T::type Constant(Index size, const _Scalar& value)
	{
		return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
			_Rows==Dynamic ? size : _Rows,
			_Columns==Dynamic ? size : _Columns, 
			_Batches, functor::ConstantFunctor<_Scalar>(value));
	}
	template<typename T = std::enable_if<_Batches != Dynamic && _Rows != Dynamic && _Columns != Dynamic,
		NullaryOp_t<functor::ConstantFunctor<_Scalar> >>>
		static T::type Constant(const _Scalar& value)
	{
		return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
			_Rows, _Columns, _Batches, functor::ConstantFunctor<_Scalar>(value));
	}

	void setZero()
	{
		Index s = size();
		if (s > 0) {
			CUMAT_SAFE_CALL(cudaMemsetAsync(data(), 0, sizeof(_Scalar) * size(), Context::current().stream()));
		}
	}

	// SLICING + BLOCKS

	//most general version, static size
	template<int NRows, int NColumns, int NBatches>
	MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, Type>
		block(Index start_row, Index start_column, Index start_batch, Index num_rows = NRows, 
            Index num_columns = NColumns, Index num_batches = NBatches)
	{
        CUMAT_ASSERT_ARGUMENT(NRows > 0 ? NRows == num_rows : true);
        CUMAT_ASSERT_ARGUMENT(NColumns > 0 ? NColumns == num_columns : true);
        CUMAT_ASSERT_ARGUMENT(NBatches > 0 ? NBatches == num_batches : true);
        CUMAT_ASSERT_ARGUMENT(num_rows >= 0);
        CUMAT_ASSERT_ARGUMENT(num_columns >= 0);
        CUMAT_ASSERT_ARGUMENT(num_batches >= 0);
		CUMAT_ASSERT_ARGUMENT(start_row >= 0);
		CUMAT_ASSERT_ARGUMENT(start_column >= 0);
		CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
		CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
		CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
		CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
		return MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, Type>(
			*this, NRows, NColumns, NBatches, start_row, start_column, start_batch);
	}
	template<int NRows, int NColumns, int NBatches>
	MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, const Type>
		block(Index start_row, Index start_column, Index start_batch, Index num_rows = NRows,
            Index num_columns = NColumns, Index num_batches = NBatches) const
	{
        CUMAT_ASSERT_ARGUMENT(NRows > 0 ? NRows == num_rows : true);
        CUMAT_ASSERT_ARGUMENT(NColumns > 0 ? NColumns == num_columns : true);
        CUMAT_ASSERT_ARGUMENT(NBatches > 0 ? NBatches == num_batches : true);
        CUMAT_ASSERT_ARGUMENT(num_rows >= 0);
        CUMAT_ASSERT_ARGUMENT(num_columns >= 0);
        CUMAT_ASSERT_ARGUMENT(num_batches >= 0);
        CUMAT_ASSERT_ARGUMENT(start_row >= 0);
        CUMAT_ASSERT_ARGUMENT(start_column >= 0);
        CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
        CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
        CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
        CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
		return MatrixBlock<_Scalar, NRows, NColumns, NBatches, _Flags, const Type>(
			*this, NRows, NColumns, NBatches, start_row, start_column, start_batch);
	}

	//most general version, dynamic size
	MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, Type>
		block(Index start_row, Index start_column, Index start_batch, Index num_rows, Index num_columns, Index num_batches)
	{
		CUMAT_ASSERT_ARGUMENT(start_row >= 0);
		CUMAT_ASSERT_ARGUMENT(start_column >= 0);
		CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
		CUMAT_ASSERT_ARGUMENT(num_rows > 0);
		CUMAT_ASSERT_ARGUMENT(num_columns > 0);
		CUMAT_ASSERT_ARGUMENT(num_batches > 0);
		CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
		CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
		CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
		return MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, Type>(
			*this, num_rows, num_columns, num_batches, start_row, start_column, start_batch);
	}
	MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, const Type>
		block(Index start_row, Index start_column, Index start_batch, Index num_rows, Index num_columns, Index num_batches) const
	{
		CUMAT_ASSERT_ARGUMENT(start_row >= 0);
		CUMAT_ASSERT_ARGUMENT(start_column >= 0);
		CUMAT_ASSERT_ARGUMENT(start_batch >= 0);
		CUMAT_ASSERT_ARGUMENT(num_rows > 0);
		CUMAT_ASSERT_ARGUMENT(num_columns > 0);
		CUMAT_ASSERT_ARGUMENT(num_batches > 0);
		CUMAT_ASSERT_ARGUMENT(start_row + num_rows <= rows());
		CUMAT_ASSERT_ARGUMENT(start_column + num_columns <= cols());
		CUMAT_ASSERT_ARGUMENT(start_batch + num_batches <= batches());
		return MatrixBlock<_Scalar, Dynamic, Dynamic, Dynamic, _Flags, const Type>(
			*this, num_rows, num_columns, num_batches, start_row, start_column, start_batch);
	}


	// TODO: specializations for batch==1, vectors, slices

};


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
CUMAT_DEF_MATRIX2(float, f)
CUMAT_DEF_MATRIX2(double, d)
CUMAT_DEF_MATRIX2(cfloat, cf)
CUMAT_DEF_MATRIX2(cdouble, cd)

#undef CUMAT_DEF_MATRIX2
#undef CUMAT_DEF_MATRIX1


CUMAT_NAMESPACE_END

#endif