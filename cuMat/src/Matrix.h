#ifndef __CUMAT_MATRIX_H__
#define __CUMAT_MATRIX_H__

#include "Macros.h"
#include "Constants.h"
#include "Context.h"
#include "DevicePointer.h"

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
		void swap(DenseStorage& other) { std::swap(data_, other.data_); }
		static __host__ __device__ CUMAT_STRONG_INLINE Index rows() { return _Rows; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index cols() { return _Columns; }
		static __host__ __device__ CUMAT_STRONG_INLINE Index batches() { return _Batches; }
		__host__ __device__ CUMAT_STRONG_INLINE const _Scalar *data() const { return data_.pointer(); }
		__host__ __device__ CUMAT_STRONG_INLINE _Scalar *data() { return data_.pointer(); }
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
 * \tparam _Scalar the scalar type of the matrix
 * \tparam _Rows the number of rows, can be a compile-time constant or cuMat::Dynamic
 * \tparam _Columns the number of cols, can be a compile-time constant or Dynamic
 * \tparam _Batches the number of batches, can be a compile-time constant or Dynamic
 * \tparam _Flags a combination of flags from the \ref Flags enum.
 */
template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
class Matrix
{
protected:
	internal::DenseStorage<_Scalar, _Rows, _Columns, _Batches> data_;
public:
	enum { Flags = _Flags };

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

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return data_.rows(); }

	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return data_.cols(); }

	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return data_.batches(); }

	__host__ __device__ CUMAT_STRONG_INLINE Index size() const { return rows()*cols()*batches(); }

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

	__device__ CUMAT_STRONG_INLINE _Scalar& rawCoeff(Index index)
	{
		CUMAT_ASSERT_CUDA(index >= 0);
		CUMAT_ASSERT_CUDA(index < size());
		return data_.data()[index];
	}

	__device__ CUMAT_STRONG_INLINE const _Scalar& rawCoeff(Index index) const
	{
		CUMAT_ASSERT_CUDA(index >= 0);
		CUMAT_ASSERT_CUDA(index < size());
		return data_.data()[index];
	}

	__host__ __device__ CUMAT_STRONG_INLINE _Scalar* data()
	{
		return data_.data();
	}

	__host__ __device__ CUMAT_STRONG_INLINE const _Scalar* data() const
	{
		return data_.data();
	}

	/**
	 * \brief Performs a sychronous copy from host data into the
	 * device memory of this matrix.
	 * 
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
	*
	* \param data the data in which the matrix is stored
	*/
	void copyToHost(_Scalar* data) const
	{
		CUMAT_SAFE_CALL(cudaMemcpy(data, data_.data(), sizeof(_Scalar)*size(), cudaMemcpyDeviceToHost));
	}
};

CUMAT_NAMESPACE_END

#endif