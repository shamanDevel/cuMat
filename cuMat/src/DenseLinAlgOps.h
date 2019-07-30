#ifndef __CUMAT_DENSE_LIN_ALG_OPS_H__
#define __CUMAT_DENSE_LIN_ALG_OPS_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "Constants.h"
#include "Context.h"
#include "MatrixBase.h"

/*
 * This file contains general expression for linear algebra operations:
 * Determinant, inverse, solver.
 * The purpose is to delegate to explicit solutions for matrices up to 4x4
 * and then to delegate to decompositions for larger matrices
 */

CUMAT_NAMESPACE_BEGIN

namespace internal
{
	namespace kernels
	{

		// MATRIX CLASS

		template<typename _Scalar, int _Dims, int _Flags>
		struct DeviceMatrix;

		template<typename _Scalar, int _Flags>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 1, _Flags>
		{
			_Scalar m00;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(batch, m00);
			}
		};

		template<typename _Scalar>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 2, Flags::RowMajor>
		{
			_Scalar m00; //1st row
			_Scalar m01;
			_Scalar m10; //2nd row
			_Scalar m11;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
				m01 = mat.coeff(0, 1, batch, -1);
				m10 = mat.coeff(1, 0, batch, -1);
				m11 = mat.coeff(1, 1, batch, -1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(0 + 4 * batch, m00);
				matrix.setRawCoeff(1 + 4 * batch, m01);
				matrix.setRawCoeff(2 + 4 * batch, m10);
				matrix.setRawCoeff(3 + 4 * batch, m11);
			}
		};
		template<typename _Scalar>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 2, Flags::ColumnMajor>
		{
			_Scalar m00; //1st column
			_Scalar m10;
			_Scalar m01; //2nd column
			_Scalar m11;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
				m10 = mat.coeff(1, 0, batch, -1);
				m01 = mat.coeff(0, 1, batch, -1);
				m11 = mat.coeff(1, 1, batch, 1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(0 + 4 * batch, m00);
				matrix.setRawCoeff(1 + 4 * batch, m10);
				matrix.setRawCoeff(2 + 4 * batch, m01);
				matrix.setRawCoeff(3 + 4 * batch, m11);
			}
		};

		template<typename _Scalar>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 3, Flags::RowMajor>
		{
			_Scalar m00; //1st row
			_Scalar m01;
			_Scalar m02;
			_Scalar m10; //2nd row
			_Scalar m11;
			_Scalar m12;
			_Scalar m20; //3rd row
			_Scalar m21;
			_Scalar m22;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
				m01 = mat.coeff(0, 1, batch, -1);
				m02 = mat.coeff(0, 2, batch, -1);
				m10 = mat.coeff(1, 0, batch, -1);
				m11 = mat.coeff(1, 1, batch, -1);
				m12 = mat.coeff(1, 2, batch, -1);
				m20 = mat.coeff(2, 0, batch, -1);
				m21 = mat.coeff(2, 1, batch, -1);
				m22 = mat.coeff(2, 2, batch, -1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(0 + 9 * batch, m00);
				matrix.setRawCoeff(1 + 9 * batch, m01);
				matrix.setRawCoeff(2 + 9 * batch, m02);
				matrix.setRawCoeff(3 + 9 * batch, m10);
				matrix.setRawCoeff(4 + 9 * batch, m11);
				matrix.setRawCoeff(5 + 9 * batch, m12);
				matrix.setRawCoeff(6 + 9 * batch, m20);
				matrix.setRawCoeff(7 + 9 * batch, m21);
				matrix.setRawCoeff(8 + 9 * batch, m22);
			}
		};
		template<typename _Scalar>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 3, Flags::ColumnMajor>
		{
			_Scalar m00; //1st column
			_Scalar m10;
			_Scalar m20;
			_Scalar m01; //2nd column
			_Scalar m11;
			_Scalar m21;
			_Scalar m02; //3rd column
			_Scalar m12;
			_Scalar m22;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
				m10 = mat.coeff(1, 0, batch, -1);
				m20 = mat.coeff(2, 0, batch, -1);
				m01 = mat.coeff(0, 1, batch, -1);
				m11 = mat.coeff(1, 1, batch, -1);
				m21 = mat.coeff(2, 1, batch, -1);
				m02 = mat.coeff(0, 2, batch, -1);
				m12 = mat.coeff(1, 2, batch, -1);
				m22 = mat.coeff(2, 2, batch, -1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(0 + 9 * batch, m00);
				matrix.setRawCoeff(1 + 9 * batch, m10);
				matrix.setRawCoeff(2 + 9 * batch, m20);
				matrix.setRawCoeff(3 + 9 * batch, m01);
				matrix.setRawCoeff(4 + 9 * batch, m11);
				matrix.setRawCoeff(5 + 9 * batch, m21);
				matrix.setRawCoeff(6 + 9 * batch, m02);
				matrix.setRawCoeff(7 + 9 * batch, m12);
				matrix.setRawCoeff(8 + 9 * batch, m22);
			}
		};

		template<typename _Scalar>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 4, Flags::RowMajor>
		{
			_Scalar m00; //1st row
			_Scalar m01;
			_Scalar m02;
			_Scalar m03;
			_Scalar m10; //2nd row
			_Scalar m11;
			_Scalar m12;
			_Scalar m13;
			_Scalar m20; //3rd row
			_Scalar m21;
			_Scalar m22;
			_Scalar m23;
			_Scalar m30; //4th row
			_Scalar m31;
			_Scalar m32;
			_Scalar m33;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
				m01 = mat.coeff(0, 1, batch, -1);
				m02 = mat.coeff(0, 2, batch, -1);
				m03 = mat.coeff(0, 3, batch, -1);
				m10 = mat.coeff(1, 0, batch, -1);
				m11 = mat.coeff(1, 1, batch, -1);
				m12 = mat.coeff(1, 2, batch, -1);
				m13 = mat.coeff(1, 3, batch, -1);
				m20 = mat.coeff(2, 0, batch, -1);
				m21 = mat.coeff(2, 1, batch, -1);
				m22 = mat.coeff(2, 2, batch, -1);
				m23 = mat.coeff(2, 3, batch, -1);
				m30 = mat.coeff(3, 0, batch, -1);
				m31 = mat.coeff(3, 1, batch, -1);
				m32 = mat.coeff(3, 2, batch, -1);
				m33 = mat.coeff(3, 3, batch, -1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(0  + 16 * batch, m00);
				matrix.setRawCoeff(1  + 16 * batch, m01);
				matrix.setRawCoeff(2  + 16 * batch, m02);
				matrix.setRawCoeff(3  + 16 * batch, m03);
				matrix.setRawCoeff(4  + 16 * batch, m10);
				matrix.setRawCoeff(5  + 16 * batch, m11);
				matrix.setRawCoeff(6  + 16 * batch, m12);
				matrix.setRawCoeff(7  + 16 * batch, m13);
				matrix.setRawCoeff(8  + 16 * batch, m20);
				matrix.setRawCoeff(9  + 16 * batch, m21);
				matrix.setRawCoeff(10 + 16 * batch, m22);
				matrix.setRawCoeff(11 + 16 * batch, m23);
				matrix.setRawCoeff(12 + 16 * batch, m30);
				matrix.setRawCoeff(13 + 16 * batch, m31);
				matrix.setRawCoeff(14 + 16 * batch, m32);
				matrix.setRawCoeff(15 + 16 * batch, m33);
			}
		};
		template<typename _Scalar>
		struct alignas(_Scalar) DeviceMatrix<_Scalar, 4, Flags::ColumnMajor>
		{
			_Scalar m00; //1st column
			_Scalar m10;
			_Scalar m20;
			_Scalar m30;
			_Scalar m01; //2nd column
			_Scalar m11;
			_Scalar m21;
			_Scalar m31;
			_Scalar m02; //3rd column
			_Scalar m12;
			_Scalar m22;
			_Scalar m32;
			_Scalar m03; //4th column
			_Scalar m13;
			_Scalar m23;
			_Scalar m33;

			template<typename T>
			__device__ CUMAT_STRONG_INLINE void load(const T& mat, Index batch)
			{
				m00 = mat.coeff(0, 0, batch, -1);
				m10 = mat.coeff(1, 0, batch, -1);
				m20 = mat.coeff(2, 0, batch, -1);
				m30 = mat.coeff(3, 0, batch, -1);
				m01 = mat.coeff(0, 1, batch, -1);
				m11 = mat.coeff(1, 1, batch, -1);
				m21 = mat.coeff(2, 1, batch, -1);
				m31 = mat.coeff(3, 1, batch, -1);
				m02 = mat.coeff(0, 2, batch, -1);
				m12 = mat.coeff(1, 2, batch, -1);
				m22 = mat.coeff(2, 2, batch, -1);
				m32 = mat.coeff(3, 2, batch, -1);
				m03 = mat.coeff(0, 3, batch, -1);
				m13 = mat.coeff(1, 3, batch, -1);
				m23 = mat.coeff(2, 3, batch, -1);
				m33 = mat.coeff(3, 3, batch, -1);
			}
			template<typename T>
			__device__ CUMAT_STRONG_INLINE void store(T& matrix, Index batch)
			{
				matrix.setRawCoeff(0  + 16 * batch, m00);
				matrix.setRawCoeff(1  + 16 * batch, m10);
				matrix.setRawCoeff(2  + 16 * batch, m20);
				matrix.setRawCoeff(3  + 16 * batch, m30);
				matrix.setRawCoeff(4  + 16 * batch, m01);
				matrix.setRawCoeff(5  + 16 * batch, m11);
				matrix.setRawCoeff(6  + 16 * batch, m21);
				matrix.setRawCoeff(7  + 16 * batch, m31);
				matrix.setRawCoeff(8  + 16 * batch, m02);
				matrix.setRawCoeff(9  + 16 * batch, m12);
				matrix.setRawCoeff(10 + 16 * batch, m22);
				matrix.setRawCoeff(11 + 16 * batch, m32);
				matrix.setRawCoeff(12 + 16 * batch, m03);
				matrix.setRawCoeff(13 + 16 * batch, m13);
				matrix.setRawCoeff(14 + 16 * batch, m23);
				matrix.setRawCoeff(15 + 16 * batch, m33);
			}
		};

		// LOAD + STORE

		template <int Dims, typename Scalar, int Rows, int Cols, int Batches, int Flags>
		__device__ CUMAT_STRONG_INLINE DeviceMatrix<Scalar, Dims, Flags> loadMat(const Matrix<Scalar, Rows, Cols, Batches, Flags>& mat, Index index)
		{
			const DeviceMatrix<Scalar, Dims, Flags>* data = reinterpret_cast<DeviceMatrix<Scalar, Dims, Flags>*>(mat.data());
			return data[index];
		}
		template <int Dims, typename T,
			typename Scalar = typename internal::traits<T>::Scalar, int Flags = internal::traits<T>::Flags>
			__device__ CUMAT_STRONG_INLINE DeviceMatrix<Scalar, Dims, Flags> loadMat(const T& mat, Index index)
		{
			DeviceMatrix<Scalar, Dims, Flags> m;
			m.load(mat, index);
			return m;
		}

		template <int Dims, typename Scalar, int Rows, int Cols, int Batches, int Flags>
		__device__ CUMAT_STRONG_INLINE void storeMat(Matrix<Scalar, Rows, Cols, Batches, Flags>& mat, const DeviceMatrix<Scalar, Dims, Flags>& out, Index index)
		{
			DeviceMatrix<Scalar, Dims, Flags>* data = reinterpret_cast<DeviceMatrix<Scalar, Dims, Flags>*>(mat.data());
			data[index] = out;
		}
		template <int Dims, typename T,
			typename Scalar = typename internal::traits<T>::Scalar, int Flags = internal::traits<T>::Flags>
			__device__ CUMAT_STRONG_INLINE void storeMat(T& mat, const DeviceMatrix<Scalar, Dims, Flags>& out, Index index)
		{
			out.store(mat, index);
		}

		// DETERMINANT

		template<typename Scalar, int Dims, typename Input>
		struct DeterminantFunctor;
		template<typename Scalar, typename Input>
		struct DeterminantFunctor<Scalar, 1, Input>
		{
			static __device__ CUMAT_STRONG_INLINE Scalar run(const Input& mat)
			{
				return mat.m00;
			}
		};
		template<typename Scalar, typename Input>
		struct DeterminantFunctor<Scalar, 2, Input>
		{
			static __device__ CUMAT_STRONG_INLINE Scalar run(const Input& mat)
			{
				return mat.m00*mat.m11 - mat.m10*mat.m01;
			}
		};
		template<typename Scalar, typename Input>
		struct DeterminantFunctor<Scalar, 3, Input>
		{
			static __device__ CUMAT_STRONG_INLINE Scalar run(const Input& mat)
			{
				return mat.m00*mat.m11*mat.m22 + mat.m01*mat.m12*mat.m20 + mat.m02*mat.m10*mat.m21
					- mat.m02*mat.m11*mat.m20 - mat.m01*mat.m10*mat.m22 - mat.m00*mat.m12*mat.m21;
			}
		};
		template<typename Scalar, typename Input>
		struct DeterminantFunctor<Scalar, 4, Input>
		{
			static __device__ CUMAT_STRONG_INLINE Scalar run(const Input& mat)
			{
				Scalar fA0 = mat.m00 * mat.m11 - mat.m01 * mat.m10;
				Scalar fA1 = mat.m00 * mat.m12 - mat.m02 * mat.m10;
				Scalar fA2 = mat.m00 * mat.m13 - mat.m03 * mat.m10;
				Scalar fA3 = mat.m01 * mat.m12 - mat.m02 * mat.m11;
				Scalar fA4 = mat.m01 * mat.m13 - mat.m03 * mat.m11;
				Scalar fA5 = mat.m02 * mat.m13 - mat.m03 * mat.m12;
				Scalar fB0 = mat.m20 * mat.m31 - mat.m21 * mat.m30;
				Scalar fB1 = mat.m20 * mat.m32 - mat.m22 * mat.m30;
				Scalar fB2 = mat.m20 * mat.m33 - mat.m23 * mat.m30;
				Scalar fB3 = mat.m21 * mat.m32 - mat.m22 * mat.m31;
				Scalar fB4 = mat.m21 * mat.m33 - mat.m23 * mat.m31;
				Scalar fB5 = mat.m22 * mat.m33 - mat.m23 * mat.m32;
				Scalar fDet = fA0 * fB5 - fA1 * fB4 + fA2 * fB3 + fA3 * fB2 - fA4 * fB1 + fA5 * fB0;
				return fDet;
			}
		};
		template <typename T, typename M, int Dims,
			typename Scalar = typename internal::traits<T>::Scalar,
			int TFlags = internal::traits<T>::Flags, int MFlags = internal::traits<M>::Flags>
			__global__ void DeterminantKernel(dim3 virtual_size, const T expr, M matrix)
		{
			CUMAT_KERNEL_1D_LOOP(index, virtual_size)
				DeviceMatrix<Scalar, Dims, TFlags> in = loadMat<Dims, T, Scalar>(expr, index);
			Scalar det = DeterminantFunctor<Scalar, Dims, DeviceMatrix<Scalar, Dims, TFlags> >::run(in);
			matrix.setRawCoeff(index, det);
			CUMAT_KERNEL_1D_LOOP_END
		}

		// INVERSE

		template<typename Scalar, int Dims, typename Input, typename Output>
		struct InverseFunctor;
		template<typename Scalar, typename Input, typename Output>
		struct InverseFunctor<Scalar, 1, Input, Output>
		{
			static __device__ CUMAT_STRONG_INLINE Output run(const Input& mat, Scalar& det)
			{
				Output o;
				o.m00 = Scalar(1)/mat.m00;
				det = mat.m00;
				return o;
			}
		};
		template<typename Scalar, typename Input, typename Output>
		struct InverseFunctor<Scalar, 2, Input, Output>
		{
			static __device__ CUMAT_STRONG_INLINE Output run(const Input& mat, Scalar& det)
			{
				det = mat.m00*mat.m11 - mat.m10*mat.m01;
				Scalar f = Scalar(1) / det;
				Output o;
				o.m00 = f * mat.m11;
				o.m10 = -f * mat.m10;
				o.m01 = -f * mat.m01;
				o.m11 = f * mat.m00;
				return o;
			}
		};
		template<typename Scalar, typename Input, typename Output>
		struct InverseFunctor<Scalar, 3, Input, Output>
		{
			static __device__ CUMAT_STRONG_INLINE Output run(const Input& mat, Scalar& det)
			{
				det = mat.m00*mat.m11*mat.m22 + mat.m01*mat.m12*mat.m20 + mat.m02*mat.m10*mat.m21
					- mat.m02*mat.m11*mat.m20 - mat.m01*mat.m10*mat.m22 - mat.m00*mat.m12*mat.m21;
				Scalar f = Scalar(1) / det;

				Output o;
				o.m00 = (mat.m11 * mat.m22 - mat.m12 * mat.m21) * f;
				o.m01 = (mat.m02 * mat.m21 - mat.m01 * mat.m22) * f;
				o.m02 = (mat.m01 * mat.m12 - mat.m02 * mat.m11) * f;
				o.m10 = (mat.m12 * mat.m20 - mat.m10 * mat.m22) * f;
				o.m11 = (mat.m00 * mat.m22 - mat.m02 * mat.m20) * f;
				o.m12 = (mat.m02 * mat.m10 - mat.m00 * mat.m12) * f;
				o.m20 = (mat.m10 * mat.m21 - mat.m11 * mat.m20) * f;
				o.m21 = (mat.m01 * mat.m20 - mat.m00 * mat.m21) * f;
				o.m22 = (mat.m00 * mat.m11 - mat.m01 * mat.m10) * f;
				return o;
			}
		};
		template<typename Scalar, typename Input, typename Output>
		struct InverseFunctor<Scalar, 4, Input, Output>
		{
			static __device__ CUMAT_STRONG_INLINE Output run(const Input& mat, Scalar& det)
			{
				Scalar fA0 = mat.m00 * mat.m11 - mat.m01 * mat.m10;
				Scalar fA1 = mat.m00 * mat.m12 - mat.m02 * mat.m10;
				Scalar fA2 = mat.m00 * mat.m13 - mat.m03 * mat.m10;
				Scalar fA3 = mat.m01 * mat.m12 - mat.m02 * mat.m11;
				Scalar fA4 = mat.m01 * mat.m13 - mat.m03 * mat.m11;
				Scalar fA5 = mat.m02 * mat.m13 - mat.m03 * mat.m12;
				Scalar fB0 = mat.m20 * mat.m31 - mat.m21 * mat.m30;
				Scalar fB1 = mat.m20 * mat.m32 - mat.m22 * mat.m30;
				Scalar fB2 = mat.m20 * mat.m33 - mat.m23 * mat.m30;
				Scalar fB3 = mat.m21 * mat.m32 - mat.m22 * mat.m31;
				Scalar fB4 = mat.m21 * mat.m33 - mat.m23 * mat.m31;
				Scalar fB5 = mat.m22 * mat.m33 - mat.m23 * mat.m32;
				det = fA0 * fB5 - fA1 * fB4 + fA2 * fB3 + fA3 * fB2 - fA4 * fB1 + fA5 * fB0;
				Scalar f = Scalar(1) / det;

				Output o;
				o.m00 = (+mat.m11 * fB5 - mat.m12 * fB4 + mat.m13 * fB3) * f;
				o.m10 = (-mat.m10 * fB5 + mat.m12 * fB2 - mat.m13 * fB1) * f;
				o.m20 = (+mat.m10 * fB4 - mat.m11 * fB2 + mat.m13 * fB0) * f;
				o.m30 = (-mat.m10 * fB3 + mat.m11 * fB1 - mat.m12 * fB0) * f;
				o.m01 = (-mat.m01 * fB5 + mat.m02 * fB4 - mat.m03 * fB3) * f;
				o.m11 = (+mat.m00 * fB5 - mat.m02 * fB2 + mat.m03 * fB1) * f;
				o.m21 = (-mat.m00 * fB4 + mat.m01 * fB2 - mat.m03 * fB0) * f;
				o.m31 = (+mat.m00 * fB3 - mat.m01 * fB1 + mat.m02 * fB0) * f;
				o.m02 = (+mat.m31 * fA5 - mat.m32 * fA4 + mat.m33 * fA3) * f;
				o.m12 = (-mat.m30 * fA5 + mat.m32 * fA2 - mat.m33 * fA1) * f;
				o.m22 = (+mat.m30 * fA4 - mat.m31 * fA2 + mat.m33 * fA0) * f;
				o.m32 = (-mat.m30 * fA3 + mat.m31 * fA1 - mat.m32 * fA0) * f;
				o.m03 = (-mat.m21 * fA5 + mat.m22 * fA4 - mat.m23 * fA3) * f;
				o.m13 = (+mat.m20 * fA5 - mat.m22 * fA2 + mat.m23 * fA1) * f;
				o.m23 = (-mat.m20 * fA4 + mat.m21 * fA2 - mat.m23 * fA0) * f;
				o.m33 = (+mat.m20 * fA3 - mat.m21 * fA1 + mat.m22 * fA0) * f;
				return o;
			}
		};
		template <typename MatIn, typename MatOut, int Dims,
			typename Scalar = typename internal::traits<MatIn>::Scalar,
			int InFlags = internal::traits<MatIn>::Flags, int OutFlags = internal::traits<MatOut>::Flags>
		__global__ void InverseKernel(dim3 virtual_size, const MatIn expr, MatOut matOut) //TODO
		{
			typedef DeviceMatrix<Scalar, Dims, InFlags> Min;
			typedef DeviceMatrix<Scalar, Dims, OutFlags> Mout;
			CUMAT_KERNEL_1D_LOOP(index, virtual_size)
				Min in = loadMat<Dims, MatIn, Scalar>(expr, index);
				Scalar det;
				Mout out = InverseFunctor<Scalar, Dims, Min, Mout >::run(in, det);
				storeMat(matOut, out, index);
			CUMAT_KERNEL_1D_LOOP_END
		}
		template <typename MatIn, typename MatOut, typename DetOut, int Dims,
			typename Scalar = typename internal::traits<MatIn>::Scalar,
			int InFlags = internal::traits<MatIn>::Flags, int OutFlags = internal::traits<MatOut>::Flags>
		__global__ void InverseKernelWithDet(dim3 virtual_size, const MatIn expr, MatOut matOut, DetOut detOut) //TODO
		{
			typedef DeviceMatrix<Scalar, Dims, InFlags> Min;
			typedef DeviceMatrix<Scalar, Dims, OutFlags> Mout;
			CUMAT_KERNEL_1D_LOOP(index, virtual_size)
				Min in = loadMat<Dims, MatIn, Scalar>(expr, index);
				Scalar det;
				Mout out = InverseFunctor<Scalar, Dims, Min, Mout >::run(in, det);
				storeMat(matOut, out, index);
				detOut.setRawCoeff(index, det);
			CUMAT_KERNEL_1D_LOOP_END
		}
	}
}


namespace internal
{
    struct DeterminantSrcTag {};
    template<typename _Child>
    struct traits<DeterminantOp<_Child> >
    {
        using Scalar = typename internal::traits<_Child>::Scalar;
        enum
        {
            Flags = internal::traits<_Child>::Flags,
            RowsAtCompileTime = 1,
            ColsAtCompileTime = 1,
            BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
            //InputSize = std::max(internal::traits<_Child>::RowsAtCompileTime, internal::traits<_Child>::ColsAtCompileTime), //std::max is not constexpr in gcc
            InputSize = internal::traits<_Child>::RowsAtCompileTime>internal::traits<_Child>::ColsAtCompileTime ? internal::traits<_Child>::RowsAtCompileTime : internal::traits<_Child>::ColsAtCompileTime,
            AccessFlags = InputSize<=4 ? ReadCwise : 0 //for matrices <=4, we also support cwise evaluation
        };
        typedef DeterminantSrcTag SrcTag;
        typedef DeletedDstTag DstTag;
    };

	struct InverseSrcTag {};
	template<typename _Child>
	struct traits<InverseOp<_Child> >
	{
		using Scalar = typename internal::traits<_Child>::Scalar;
		enum
		{
			HasExplicitFormula = internal::traits<_Child>::RowsAtCompileTime >= 1 && internal::traits<_Child>::RowsAtCompileTime <= 4,
			RequiresColumnMajor = !HasExplicitFormula, //LUDecomposition-Solve requires a Column Major output matrix, so use it if we might fallback to LU.
			Flags = RequiresColumnMajor ? CUMAT_NAMESPACE Flags::ColumnMajor : internal::traits<_Child>::Flags,
			RowsAtCompileTime = internal::traits<_Child>::RowsAtCompileTime,
			ColsAtCompileTime = internal::traits<_Child>::ColsAtCompileTime,
			BatchesAtCompileTime = internal::traits<_Child>::BatchesAtCompileTime,
			InputSize = internal::traits<_Child>::RowsAtCompileTime > internal::traits<_Child>::ColsAtCompileTime ? internal::traits<_Child>::RowsAtCompileTime : internal::traits<_Child>::ColsAtCompileTime,
			AccessFlags = 0
		};
		typedef InverseSrcTag SrcTag;
		typedef DeletedDstTag DstTag;
	};
}

template<typename _Child>
class DeterminantOp : public MatrixBase<DeterminantOp<_Child> >
{
public:
    typedef MatrixBase<DeterminantOp<_Child> > Base;
    using Scalar = typename internal::traits<_Child>::Scalar;
    using Child = _Child;
    enum
    {
        Flags = internal::traits<_Child>::Flags,
        Rows = 1,
        Columns = 1,
        Batches = internal::traits<_Child>::BatchesAtCompileTime,
        InputSize = internal::traits<_Child>::RowsAtCompileTime>internal::traits<_Child>::ColsAtCompileTime ? internal::traits<_Child>::RowsAtCompileTime : internal::traits<_Child>::ColsAtCompileTime
    };

private:
    const _Child matrix_;

public:
    explicit DeterminantOp(const MatrixBase<_Child>& matrix)
        : matrix_(matrix.derived())
    {
		CUMAT_ASSERT_DIMENSION(matrix.rows() == matrix.cols());
    }

    const _Child& getMatrix() const { return matrix_; }

    __host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return 1; }
    __host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return 1; }
    __host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }

    template<int Dims = InputSize>
    __device__ CUMAT_STRONG_INLINE Scalar coeff(Index row, Index col, Index batch, Index index) const
    {
        CUMAT_STATIC_ASSERT(Dims <= 4, "Cwise-evaluation of the determinant is only supported for matrices of size <= 4x4");
        typedef internal::kernels::DeviceMatrix<Scalar, Dims, internal::traits<_Child>::Flags> Mat_t;
        Mat_t in = internal::kernels::loadMat<Dims, _Child, Scalar>(matrix_, batch);
        Scalar det = internal::kernels::DeterminantFunctor<Scalar, Dims, Mat_t>::run(in);
        return det;
    }
};

template<typename _Child>
class InverseOp : public MatrixBase<InverseOp<_Child> >
{
public:
	typedef MatrixBase<InverseOp<_Child> > Base;
	using Scalar = typename internal::traits<_Child>::Scalar;
	using Child = _Child;
	enum
	{
		Flags = internal::traits<_Child>::Flags,
		Rows = internal::traits<_Child>::RowsAtCompileTime,
		Columns = internal::traits<_Child>::ColsAtCompileTime,
		Batches = internal::traits<_Child>::BatchesAtCompileTime,
		InputSize = internal::traits<_Child>::RowsAtCompileTime > internal::traits<_Child>::ColsAtCompileTime ? internal::traits<_Child>::RowsAtCompileTime : internal::traits<_Child>::ColsAtCompileTime
	};

private:
	const _Child matrix_;

public:
	explicit InverseOp(const MatrixBase<_Child>& matrix)
		: matrix_(matrix.derived())
	{
		CUMAT_ASSERT_DIMENSION(matrix.rows() == matrix.cols());
	}

	const _Child& getMatrix() const { return matrix_; }

	__host__ __device__ CUMAT_STRONG_INLINE Index rows() const { return matrix_.rows(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index cols() const { return matrix_.cols(); }
	__host__ __device__ CUMAT_STRONG_INLINE Index batches() const { return matrix_.batches(); }
};

namespace internal
{
#if CUMAT_NVCC==1
    template<typename _Dst, typename _Src, AssignmentMode _AssignmentMode, typename _DstTag>
    struct Assignment<_Dst, _Src, _AssignmentMode, _DstTag, DeterminantSrcTag>
    {
    private:
        using Op = typename _Src::Type;
        using Child = typename Op::Child;

        template<typename Derived>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 1>)
        {
            //we need at least cwise-read
            typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
            Child_wrapped inWrapped(op.getMatrix());

            //launch kernel
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 1>);
			internal::kernels::DeterminantKernel<Child_wrapped, Derived, 1> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
            CUMAT_CHECK_ERROR();
        }

        template<typename Derived>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 2>)
        {
            //we need at least cwise-read
            typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
            Child_wrapped inWrapped(op.getMatrix());

            //launch kernel
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 2>);
			internal::kernels::DeterminantKernel<Child_wrapped, Derived, 2> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
            CUMAT_CHECK_ERROR();
        }

        template<typename Derived>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 3>)
        {
            //we need at least cwise-read
            typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
            Child_wrapped inWrapped(op.getMatrix());

            //launch kernel
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 3>);
			internal::kernels::DeterminantKernel<Child_wrapped, Derived, 3> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
            CUMAT_CHECK_ERROR();
        }

		template<typename Derived>
		static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 4>)
		{
			//we need at least cwise-read
			typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
			Child_wrapped inWrapped(op.getMatrix());

			//launch kernel
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 4>);
			internal::kernels::DeterminantKernel<Child_wrapped, Derived, 4> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (cfg.virtual_size, inWrapped, m);
			CUMAT_CHECK_ERROR();
		}

        template<typename Derived, int DynamicSize>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, DynamicSize>)
        {
            Index size = op.getMatrix().rows();
            CUMAT_ASSERT(op.rows() == op.cols());

            if (size <= 4)
            {
                //now we need to evaluate the input to direct read
                typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
                Child_wrapped inWrapped(op.getMatrix());
                //short-cuts
                Context& ctx = Context::current();
                if (size == 1)
                {
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 1>);
					internal::kernels::DeterminantKernel<Child_wrapped, Derived, 1> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
                }
                else if (size == 2)
                {
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 2>);
					internal::kernels::DeterminantKernel<Child_wrapped, Derived, 2> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
                }
                else if (size == 3)
                {
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 3>);
					internal::kernels::DeterminantKernel<Child_wrapped, Derived, 3> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
                }
				else if (size == 4)
				{
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::DeterminantKernel<Child_wrapped, Derived, 4>);
					internal::kernels::DeterminantKernel<Child_wrapped, Derived, 4> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (cfg.virtual_size, inWrapped, m);
				}
				CUMAT_CHECK_ERROR();
            }
            else
            {
                //use LU Decomposition
                //CUMAT_SAFE_CALL(cudaDeviceSynchronize());
                LUDecomposition<Child> lu(op.getMatrix());
                typedef typename LUDecomposition<Child>::DeterminantMatrix DetType;
                internal::Assignment<Derived, DetType, AssignmentMode::ASSIGN, typename Derived::DstTag, typename DetType::SrcTag>
                    ::assign(m.derived(), lu.determinant());
                //CUMAT_SAFE_CALL(cudaDeviceSynchronize());
            }
        }

    public:
        static void assign(_Dst& dst, const _Src& src)
        {
            static_assert(_AssignmentMode == AssignmentMode::ASSIGN, "Currently only AssignmentMode::ASSIGN is supported");
            evalImpl(src.derived(), dst.derived(), std::integral_constant<int, Op::InputSize>());
        }
    };

	template<typename _Dst, typename _Src, AssignmentMode _AssignmentMode, typename _DstTag>
    struct Assignment<_Dst, _Src, _AssignmentMode, _DstTag, InverseSrcTag>
    {
    private:
        using Op = typename _Src::Type;
        using Child = typename Op::Child;

        template<typename Derived>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 1>)
        {
            //we need at least cwise-read
            typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
            Child_wrapped inWrapped(op.getMatrix());

            //launch kernel
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 1>);
			internal::kernels::InverseKernel<Child_wrapped, Derived, 1> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
            CUMAT_CHECK_ERROR();
        }

        template<typename Derived>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 2>)
        {
            //we need at least cwise-read
            typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
            Child_wrapped inWrapped(op.getMatrix());

            //launch kernel
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 2>);
			internal::kernels::InverseKernel<Child_wrapped, Derived, 2> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
            CUMAT_CHECK_ERROR();
        }

        template<typename Derived>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 3>)
        {
            //we need at least cwise-read
            typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
            Child_wrapped inWrapped(op.getMatrix());

            //launch kernel
            Context& ctx = Context::current();
            KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 3>);
			internal::kernels::InverseKernel<Child_wrapped, Derived, 3> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
            CUMAT_CHECK_ERROR();
        }

		template<typename Derived>
		static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, 4>)
		{
			//we need at least cwise-read
			typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
			Child_wrapped inWrapped(op.getMatrix());

			//launch kernel
			Context& ctx = Context::current();
			KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 4>);
			internal::kernels::InverseKernel<Child_wrapped, Derived, 4> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (cfg.virtual_size, inWrapped, m);
			CUMAT_CHECK_ERROR();
		}

        template<typename Derived, int DynamicSize>
        static void evalImpl(const Op& op, Derived& m, std::integral_constant<int, DynamicSize>)
        {
            const Index size = op.getMatrix().rows();
            CUMAT_ASSERT_DIMENSION(op.rows() == op.cols());
			CUMAT_ASSERT_DIMENSION(op.rows() == m.rows());
			CUMAT_ASSERT_DIMENSION(op.cols() == m.cols());

            if (size <= 4)
            {
                //now we need to evaluate the input to direct read
                typedef typename MatrixReadWrapper<Child, ReadCwise>::type Child_wrapped;
                Child_wrapped inWrapped(op.getMatrix());
                //short-cuts
                Context& ctx = Context::current();
                if (size == 1)
                {
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 1>);
					internal::kernels::InverseKernel<Child_wrapped, Derived, 1> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
                }
                else if (size == 2)
                {
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 2>);
					internal::kernels::InverseKernel<Child_wrapped, Derived, 2> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
                }
                else if (size == 3)
                {
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 3>);
					internal::kernels::InverseKernel<Child_wrapped, Derived, 3> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>(cfg.virtual_size, inWrapped, m);
                }
				else if (size == 4)
				{
					KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.batches(), internal::kernels::InverseKernel<Child_wrapped, Derived, 4>);
					internal::kernels::InverseKernel<Child_wrapped, Derived, 4> <<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>> (cfg.virtual_size, inWrapped, m);
				}
				CUMAT_CHECK_ERROR();
            }
            else
            {
                //use LU Decomposition
                //CUMAT_SAFE_CALL(cudaDeviceSynchronize());
                LUDecomposition<Child> lu(op.getMatrix());
                typedef typename LUDecomposition<Child>::InverseResultType InverseType;
                internal::Assignment<Derived, InverseType, AssignmentMode::ASSIGN, typename Derived::DstTag, typename InverseType::SrcTag>
                    ::assign(m.derived(), lu.inverse());
                //CUMAT_SAFE_CALL(cudaDeviceSynchronize());
            }
        }

    public:
        static void assign(_Dst& dst, const _Src& src)
        {
            static_assert(_AssignmentMode == AssignmentMode::ASSIGN, "Currently only AssignmentMode::ASSIGN is supported");
            evalImpl(src.derived(), dst.derived(), std::integral_constant<int, Op::InputSize>());
        }
    };
#endif
}

#if CUMAT_NVCC==1
template<typename Derived, int Dims, typename InverseType, typename DetType>
struct ComputeInverseWithDet
{
	static void run(const MatrixBase<Derived>& input, InverseType& invOut, DetType& detOut)
	{
		Context& ctx = Context::current();
		using MatIn = typename Derived::Type;
		using MatOut = typename InverseType::Type;
		using DetOut = typename DetType::Type;
		KernelLaunchConfig cfg = ctx.createLaunchConfig1D(input.batches(), internal::kernels::InverseKernelWithDet<MatIn, MatOut, DetOut, Dims>);
		internal::kernels::InverseKernelWithDet<MatIn, MatOut, DetOut, Dims>
			<<<cfg.block_count, cfg.thread_per_block, 0, ctx.stream() >>>
			(cfg.virtual_size, input.derived(), invOut.derived(), detOut.derived());
		CUMAT_CHECK_ERROR();
	}
};
#endif

CUMAT_NAMESPACE_END


#endif
