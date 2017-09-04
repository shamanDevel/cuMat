#ifndef __CUMAT_MATRIX_H__
#define __CUMAT_MATRIX_H__

#include "Macros.h"
#include "Constants.h"
#include "Context.h"
#include "DevicePointer.h"
#include "MatrixBase.h"

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
		DenseStorage(const DenseStorage& other) : data_(other.data_), rows_(other.rows_) {}
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
		DenseStorage(const DenseStorage& other) : data_(other.data_), cols_(other.cols_) {}
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
		DenseStorage(const DenseStorage& other) : data_(other.data_), batches_(other.batches_) {}
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
		DenseStorage(const DenseStorage& other) : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}
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
		DenseStorage(const DenseStorage& other) : data_(other.data_), rows_(other.rows_), batches_(other.batches_) {}
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
		DenseStorage(const DenseStorage& other) : data_(other.data_), cols_(other.cols_), batches_(other.batches_) {}
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
		DenseStorage(const DenseStorage& other) 
			: data_(other.data_)
			, rows_(other.rows_)
			, cols_(other.cols_)
			, batches_(other.batches_)
		{}
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
}

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
class Matrix : public MatrixBase<Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags> >
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

	/**
	 * \brief Default constructor.
	 * For completely fixed-size matrices, this creates a matrix of that size.
	 * For (fully or partially) dynamic matrices, creates a matrix of size 0.
	 */
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
	 * \brief Constructs a matrix
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

	/**
	 * \brief Returns the total number of entries in this matrix.
	 * This value is computed as \code rows()*cols()*batches()* \endcode
	 * \return the total number of entries
	 */
	__host__ __device__ CUMAT_STRONG_INLINE Index size() const { return rows()*cols()*batches(); }

	// COEFFICIENT ACCESS

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
		CUMAT_ASSERT_CUDA(row >= 0);
		CUMAT_ASSERT_CUDA(row < rows());
		CUMAT_ASSERT_CUDA(col >= 0);
		CUMAT_ASSERT_CUDA(col < cols());
		CUMAT_ASSERT_CUDA(batch >= 0);
		CUMAT_ASSERT_CUDA(batch < batches());
		if (CUMAT_IS_ROW_MAJOR(Flags))
			return data_.data()[col + cols() * (row + rows() * batch)];
		else
			return data_.data()[row + rows() * (col + cols() * batch)];
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
		CUMAT_ASSERT_CUDA(row >= 0);
		CUMAT_ASSERT_CUDA(row < rows());
		CUMAT_ASSERT_CUDA(col >= 0);
		CUMAT_ASSERT_CUDA(col < cols());
		CUMAT_ASSERT_CUDA(batch >= 0);
		CUMAT_ASSERT_CUDA(batch < batches());
		if (CUMAT_IS_ROW_MAJOR(Flags))
			return data_.data()[col + cols() * (row + rows() * batch)];
		else
			return data_.data()[row + rows() * (col + cols() * batch)];
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
		return data_.data()[index];
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
	template<typename T = std::enable_if<_Batches == 1 || _Batches == Dynamic, EigenMatrix_t>>
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
		m.copyFromHost(mat.data());
		return m;
	}
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

#endif
};

CUMAT_NAMESPACE_END

#endif