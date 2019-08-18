#ifndef __CUMAT_SIMPLE_RANDOM_H__
#define __CUMAT_SIMPLE_RANDOM_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "DevicePointer.h"
#include "MatrixBase.h"

#include <random>
#include <chrono>
#include <vector>
#include <limits>

CUMAT_NAMESPACE_BEGIN

namespace internal
{
	namespace kernels
	{
		typedef unsigned long long state_t;

		struct RandNextHelper {
			static __device__ int eval(int bits, state_t* seed)
			{
				*seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1LL << 48) - 1);
				return (int)(*seed >> (48 - bits));
			}
		};

		template<typename S>
		struct RandNext {
			static __device__ S eval(state_t* seed, S min, S max);
		};

		template<>
		struct RandNext<int> {
			static __device__ int eval(state_t* seed, int min, int max)
			{
				int n = max - min;
				if (n <= 0)
					return 0;

				if ((n & -n) == n)  // i.e., n is a power of 2
					return (int)((n * (long)RandNextHelper::eval(31, seed)) >> 31) + min;

				int bits, val;
				do {
					bits = RandNextHelper::eval(31, seed);
					val = bits % n;
				} while (bits - val + (n - 1) < 0);
				return val + min;
			}
		};

		template<>
		struct RandNext<long long> {
			static __device__ long long eval(state_t* seed, long long min, long long max)
			{
				long long v = ((long long)(RandNextHelper::eval(32, seed)) << 32) + RandNextHelper::eval(32, seed);
				return (v % (max - min)) + min;
			}
		};

		template<>
		struct RandNext<bool> {
			static __device__ bool eval(state_t* seed, bool dummyMin, bool dummyMax)
			{
				return RandNextHelper::eval(1, seed) != 0;
			}
		};

		template<>
		struct RandNext<float> {
			static __device__ float eval(state_t* seed, float min, float max)
			{
				float v = RandNextHelper::eval(24, seed) / ((float)(1 << 24));
				return (v * (max - min)) + min;
			}
		};

		template<>
		struct RandNext<double> {
			static __device__ double eval(state_t* seed, double min, double max)
			{
				double v = (((long)(RandNextHelper::eval(26, seed)) << 27) + RandNextHelper::eval(27, seed))
					/ (double)(1LL << 53);
				return (v * (max - min)) + min;
			}
		};

		template<>
		struct RandNext<cfloat> {
			static __device__ cfloat eval(state_t* seed, cfloat min, cfloat max)
			{
				float real = RandNext<float>::eval(seed, min.real(), max.real());
				float imag = RandNext<float>::eval(seed, min.imag(), max.imag());
				return cfloat(real, imag);
			}
		};

		template<>
		struct RandNext<cdouble> {
			static __device__ cdouble eval(state_t* seed, cdouble min, cdouble max)
			{
				double real = RandNext<double>::eval(seed, min.real(), max.real());
				double imag = RandNext<double>::eval(seed, min.imag(), max.imag());
				return cdouble(real, imag);
			}
		};;

		template<typename M, typename S>
		__global__ void RandomEvaluationKernel(dim3 virtual_size, M matrix, S min, S max, state_t* seeds)
		{
			state_t seed = seeds[threadIdx.x];
			CUMAT_KERNEL_1D_LOOP(index, virtual_size)
				Index i, j, k;
			matrix.index(index, i, j, k);
			//printf("eval at row=%d, col=%d, batch=%d, index=%d\n", (int)i, (int)j, (int)k, (int)matrix.index(i, j, k));
			matrix.setRawCoeff(index, RandNext<S>::eval(&seed, min, max));
			CUMAT_KERNEL_1D_LOOP_END
				seeds[threadIdx.x] = seed;
		}

	}

	template<typename S>
	struct MinMaxDefaults;

	template<> struct MinMaxDefaults<int>
	{
		constexpr static int min() { return 0; }
		constexpr static int max() { return std::numeric_limits<int>::max(); }
	};
	template<> struct MinMaxDefaults<long>
	{
		constexpr static long min() { return 0; }
		constexpr static long max() { return std::numeric_limits<long>::max(); }
	};
	template<> struct MinMaxDefaults<long long>
	{
		constexpr static long long min() { return 0; }
		constexpr static long long max() { return std::numeric_limits<long long>::max(); }
	};
	template<> struct MinMaxDefaults<bool>
	{
		constexpr static bool min() { return false; }
		constexpr static bool max() { return true; }
	};
	template<> struct MinMaxDefaults<float>
	{
		constexpr static float min() { return 0; }
		constexpr static float max() { return 1; }
	};
	template<> struct MinMaxDefaults<double>
	{
		constexpr static double min() { return 0; }
		constexpr static double max() { return 1; }
	};
	template<> struct MinMaxDefaults<cfloat>
	{
		const static cfloat min() { return cfloat(0, 0); }
		const static cfloat max() { return cfloat(1, 1); }
	};
	template<> struct MinMaxDefaults<cdouble>
	{
		const static cdouble min() { return cdouble(0, 0); }
		const static cdouble max() { return cdouble(1, 1); }
	};
}

/**
 * \brief Utility class to create simple random numbers.
 * This class uses a simplistic pseudorandom generator. Useful for testing, but not if high-quality random numbers are required.
 * The generator is deterministic given the same initial seed and sequence of method calls.
 * 
 * Because the random number generator contains a state, it can't be implemented as a nullary op.
 * Instead, a writable matrix type (Matrix, MatrixBlock, ...) has to be passed to
 * \ref fillUniform(const MatrixBase<_Derived>& m, const _Scalar& min, const _Scalar& max) 
 * and this method fills the specified matrix.
 */
class SimpleRandom
{
private:
    static const size_t numStates = 1024;
    DevicePointer<internal::kernels::state_t> state_;

public:
    /**
     * \brief Creates a new random-number generator using the specified seed
     * \param seed the seed
     */
    SimpleRandom(unsigned int seed)
        : state_(numStates)
    {
        std::default_random_engine rnd(seed);
        std::uniform_int_distribution<internal::kernels::state_t> distr;
        std::vector<unsigned long long> v(numStates);
        for (size_t i=0; i<numStates; ++i)
        {
            v[i] = distr(rnd);
        }
        CUMAT_SAFE_CALL(cudaMemcpyAsync(state_.pointer(), v.data(), 
            numStates * sizeof(internal::kernels::state_t), cudaMemcpyHostToDevice, Context::current().stream()));
    }

    /**
     * \brief Creates a new random number generator using the current time as seed.
     */
    SimpleRandom()
        : SimpleRandom(static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()))
    {}

    /**
     * \brief Fills the specified matrix-like type with random numbers.
     * The matrix must be component-wise writable, i.e. a Matrix or MatrixBlock.
     * 
     * The matrix is then filled with random numbers between <code>min</code> (inclusive) and <code>max</code> (exclusive).
     * Generators are implemented for the types int, long, float, double and bool.
     * Bool ignores the min and max parameters.
     * 
     * \tparam _Derived the matrix type
     * \tparam _Scalar the scalar type, automatically derived from the matrix
     * \param m the matrix
     * \param min the minimal value
     * \param max the maximal value
     */
    template<
        typename _Derived,
        typename _Scalar = typename internal::traits<_Derived>::Scalar >
    void fillUniform(MatrixBase<_Derived>& m, const _Scalar& min = internal::MinMaxDefaults<_Scalar>::min(), const _Scalar& max = internal::MinMaxDefaults<_Scalar>::max())
    {
#if CUMAT_NVCC==1
        if (m.size() == 0) return;
		typedef typename _Derived::Type ActualType;
        Context& ctx = Context::current();
        KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.size(), internal::kernels::RandomEvaluationKernel<ActualType, _Scalar>);
		internal::kernels::RandomEvaluationKernel<ActualType, _Scalar> <<<1, cfg.thread_per_block.x, 0, ctx.stream() >>>(cfg.virtual_size, m.derived(), min, max, state_.pointer());
        CUMAT_CHECK_ERROR();
#else
		CUMAT_ERROR_IF_NO_NVCC(fillUniform);
#endif
    }
};

CUMAT_NAMESPACE_END

#endif