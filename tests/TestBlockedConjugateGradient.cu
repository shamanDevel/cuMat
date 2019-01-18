#include <catch/catch.hpp>

#include <cuMat/Core>
#include <cuMat/Sparse>
#include <cuMat/IterativeLinearSolvers>

#include "Utils.h"

//---------------------------------
// SPECIFY real3x3 block-matrix
//---------------------------------
typedef float real;

//real3
typedef float4 real3;
CUMAT_STRONG_INLINE __host__ __device__ real3 make_real3(real x, real y, real z) { return make_float4(x, y, z, 0); }
inline __host__ __device__ float dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
typedef cuMat::Matrix<real3, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> Vector3X;

//real3x3
/**
* \brief Row-major 3x3 matrix.
* The three rows are stored using real3 (=build-in float4).
*/
struct real3x3
{
	real3 r1, r2, r3;

	__host__ __device__ CUMAT_STRONG_INLINE real3x3() {}
	explicit __host__ __device__ CUMAT_STRONG_INLINE real3x3(real v) : r1(make_real3(v,v,v)), r2(make_real3(v, v, v)), r3(make_real3(v, v, v)) {}
	__host__ __device__ CUMAT_STRONG_INLINE real3x3(const real3& r1, const real3& r2, const real3& r3) : r1(r1), r2(r2), r3(r3) {}

	__host__ __device__ CUMAT_STRONG_INLINE real3x3 operator+(const real3x3& other) const
	{
		return real3x3(r1 + other.r1, r2 + other.r2, r3 + other.r3);
	}

	__host__ __device__ CUMAT_STRONG_INLINE real det() const
	{
		return (r1.x*r2.y*r3.z + r1.y*r2.z*r3.x + r1.z*r2.x*r3.y - r1.z*r2.y*r3.x - r1.x*r2.z*r3.y - r1.y*r2.x*r3.z);
	}

	__host__ __device__ CUMAT_STRONG_INLINE real3x3 inverse() const
	{
		real detInv = real(1) / (r1.x*r2.y*r3.z + r1.y*r2.z*r3.x + r1.z*r2.x*r3.y - r1.z*r2.y*r3.x - r1.x*r2.z*r3.y - r1.y*r2.x*r3.z);
		return real3x3(
			detInv * make_real3(r2.y*r3.z - r2.z*r3.y, r1.z*r3.y - r1.y*r3.z, r1.y*r2.z - r1.z*r2.y),
			detInv * make_real3(r2.z*r3.x - r2.x*r3.z, r1.x*r3.z - r1.z*r3.x, r1.z*r2.x - r1.x*r2.z),
			detInv * make_real3(r2.x*r3.y - r2.y*r3.x, r1.y*r3.x - r1.x*r3.y, r1.x*r2.y - r1.y*r2.x));
	}

	__host__ __device__ CUMAT_STRONG_INLINE real3x3 transpose() const
	{
		return real3x3(
			make_real3(r1.x, r2.x, r3.x),
			make_real3(r1.y, r2.y, r3.y),
			make_real3(r1.z, r2.z, r3.z));
	}
};
typedef cuMat::SparseMatrix<real3x3, 1, cuMat::CSR> SMatrix3x3;

__host__ __device__ CUMAT_STRONG_INLINE real3x3 operator*(real scalar, const real3x3& mat)
{
	return real3x3(scalar*mat.r1, scalar*mat.r2, scalar*mat.r3);
}
__host__ __device__ CUMAT_STRONG_INLINE real3x3 operator*(const real3x3& mat, real scalar)
{
	return real3x3(scalar*mat.r1, scalar*mat.r2, scalar*mat.r3);
}

//---------------------------------
// CuMat functor specializations
//---------------------------------
CUMAT_NAMESPACE_BEGIN

namespace internal
{
	//num traits
	template <>
	struct NumTraits<real3>
	{
		typedef real3 Type;
		typedef real3 RealType;
		typedef real ElementalType;
		enum
		{
			IsCudaNumeric = false,
			IsComplex = false,
		};
		static constexpr CUMAT_STRONG_INLINE real epsilon() { return std::numeric_limits<real>::epsilon(); }
	};
	template <>
	struct NumTraits<real3x3>
	{
		typedef real3x3 Type;
		typedef real3x3 RealType;
		typedef real ElementalType;
		enum
		{
			IsCudaNumeric = false,
			IsComplex = false,
		};
		static constexpr CUMAT_STRONG_INLINE real epsilon() { return std::numeric_limits<real>::epsilon(); }
	};

	//real3x3 * real3 product
	template<>
	struct ProductElementFunctor<real3x3, real3,
		ProductArgOp::NONE, ProductArgOp::NONE, ProductArgOp::NONE>
	{
		using Scalar = real3;
		static CUMAT_STRONG_INLINE __device__ Scalar mult(
			const real3x3& left, const real3& right)
		{
			return make_real3(
				dot(left.r1, right),
				dot(left.r2, right),
				dot(left.r3, right));
		}
	};

	//real3 -> diagonal real3x3
	template<>
	struct AsDiagonalFunctor<real3>
	{
		using MatrixType = real3x3;
		static __host__ __device__ CUMAT_STRONG_INLINE MatrixType asDiagonal(const real3& v)
		{
			return real3x3(
				make_real3(v.x, 0, 0),
				make_real3(0, v.y, 0),
				make_real3(0, 0, v.z)
			);
		}
	};

	//diagonal real3x3 -> real3
	template<>
	struct ExtractDiagonalFunctor<real3x3>
	{
		using VectorType = real3;
		static __host__ __device__ CUMAT_STRONG_INLINE VectorType extractDiagonal(const real3x3& v)
		{
			return make_real3(v.r1.x, v.r2.y, v.r3.z);
		}
	};
}

namespace functor
{
	//real -> real3 broadcasting
	template<>
	struct CastFunctor<real, real3>
	{
		static __device__ CUMAT_STRONG_INLINE real3 cast(const real& source)
		{
			return make_real3(source, source, source);
		}
	};

	//checked inversion of real3
	template<>
	struct UnaryMathFunctor_cwiseInverseCheck<real3>
	{
	public:
		typedef real3 ReturnType;
		__device__ CUMAT_STRONG_INLINE real3 operator()(const real3& v, Index row, Index col, Index batch) const
		{
			return make_real3(
				(v.x == real(0) ? real(1) : real(1) / v.x),
				(v.y == real(0) ? real(1) : real(1) / v.y),
				(v.z == real(0) ? real(1) : real(1) / v.z)
			);
		}
	};

	//cwiseAbs2 (for squaredNorm), must return real
	template<>
	struct UnaryMathFunctor_cwiseAbs2<real3>
	{
	public:
		typedef real ReturnType;
		__device__ CUMAT_STRONG_INLINE real operator()(const real3& v, Index row, Index col, Index batch) const
		{
			return v.x*v.x + v.y*v.y + v.z*v.z;
		}
	};

	//cwiseDot
	template<>
	struct BinaryMathFunctor_cwiseDot<real3>
	{
	public:
		typedef real ReturnType;
		__device__ CUMAT_STRONG_INLINE real operator()(const real3& x, const real3& y, Index row, Index col, Index batch) const
		{
			return dot(x, y);
		}
	};
}

CUMAT_NAMESPACE_END

//---------------------------------
// Main test case: blocked poisson problem
//---------------------------------
TEST_CASE("Blocked CG", "[CG]")
{
	//setup matrix
	typedef cuMat::SparsityPattern<cuMat::CSR> SPattern;
	SPattern pattern;
	pattern.rows = 3;
	pattern.cols = 3;
	pattern.nnz = 7;
	pattern.JA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(4) << 0, 2, 5, 7).finished());
	pattern.IA = SPattern::IndexVector::fromEigen((Eigen::VectorXi(7) << 0, 1, 0, 1, 2, 1, 2).finished());
	SMatrix3x3 mat(pattern);
	std::vector<real3x3> data1 = {
		real3x3(make_real3(2, -1, 0), make_real3(-1, 2, -1), make_real3(0, -1, 2)), real3x3(make_real3(0,0,0), make_real3(0,0,0), make_real3(-1,0,0)),
		real3x3(make_real3(0,0,-1), make_real3(0,0,0), make_real3(0,0,0)), real3x3(make_real3(2, -1, 0), make_real3(-1, 2, -1), make_real3(0, -1, 2)), real3x3(make_real3(0,0,0), make_real3(0,0,0), make_real3(-1,0,0)),
		real3x3(make_real3(0,0,-1), make_real3(0,0,0), make_real3(0,0,0)), real3x3(make_real3(2, -1, 0), make_real3(-1, 2, -1), make_real3(0, -1, 1))
	};
	mat.getData().copyFromHost(data1.data());

	//setup right hand side
	Vector3X b(3);
	std::vector<real3> data2 = { make_real3(0.1, 0, 0), make_real3(0,0,0), make_real3(0,0,-0.1) };
	b.copyFromHost(data2.data());

	//solve
	cuMat::ConjugateGradient<SMatrix3x3> cg(mat);
	cg.setMaxIterations(30);
	real tol = 1e-5;
	cg.setTolerance(tol);
	Vector3X x = cg.solve(b);
	REQUIRE(cg.error() < cg.tolerance()); //converged
	REQUIRE(cg.iterations() < cg.maxIterations());
	REQUIRE(cg.iterations() > 0); //but something was done
}