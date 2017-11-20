#ifndef __CUMAT_SIMPLE_RANDOM_H__
#define __CUMAT_SIMPLE_RANDOM_H__

#include "Macros.h"
#include "ForwardDeclarations.h"
#include "DevicePointer.h"
#include "MatrixBase.h"

#include <random>
#include <chrono>
#include <vector>

CUMAT_NAMESPACE_BEGIN

namespace
{
    typedef unsigned long long state_t;

    __device__ int randNext(int bits, state_t* seed)
    {
        *seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1LL << 48) - 1);
        return (int)(*seed >> (48 - bits));
    }

    template<typename S>
    __device__ S randNext(state_t* seed, S min, S max);

    template<>
    __device__ int randNext<int>(state_t* seed, int min, int max)
    {
        int n = max - min;
        if (n <= 0)
            return 0;

        if ((n & -n) == n)  // i.e., n is a power of 2
            return (int)((n * (long)randNext(31, seed)) >> 31) + min;

        int bits, val;
        do {
            bits = randNext(31, seed);
            val = bits % n;
        } while (bits - val + (n - 1) < 0);
        return val + min;
    }

    template<>
    __device__ long long randNext<long long>(state_t* seed, long long min, long long max)
    {
        long long v = ((long long)(randNext(32, seed)) << 32) + randNext(32, seed);
        return (v % (max - min)) + min;
    }

    template<>
    __device__ bool randNext<bool>(state_t* seed, bool dummyMin, bool dummyMax)
    {
        return randNext(1, seed) != 0;
    }

    template<>
    __device__ float randNext<float>(state_t* seed, float min, float max)
    {
        float v = randNext(24, seed) / ((float)(1 << 24));
        return (v * (max - min)) + min;
    }

    template<>
    __device__ double randNext<double>(state_t* seed, double min, double max)
    {
        double v = (((long)(randNext(26, seed)) << 27) + randNext(27, seed))
            / (double)(1LL << 53);
        return (v * (max - min)) + min;
    }

    template<typename M, typename S>
    __global__ void EvaluationKernel(dim3 virtual_size, M matrix, S min, S max, state_t* seeds)
    {
        state_t seed = seeds[threadIdx.x];
        CUMAT_KERNEL_1D_LOOP(index, virtual_size) {
            Index i, j, k;
            matrix.index(index, i, j, k);
            //printf("eval at row=%d, col=%d, batch=%d, index=%d\n", (int)i, (int)j, (int)k, (int)matrix.index(i, j, k));
            matrix.rawCoeff(index) = randNext<S>(&seed, min, max);
        }
        seeds[threadIdx.x] = seed;
    }
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
    DevicePointer<state_t> state_;

public:
    /**
     * \brief Creates a new random-number generator using the specified seed
     * \param seed the seed
     */
    SimpleRandom(unsigned int seed)
        : state_(numStates)
    {
        std::default_random_engine rnd(seed);
        std::uniform_int_distribution<state_t> distr;
        std::vector<unsigned long long> v(numStates);
        for (size_t i=0; i<numStates; ++i)
        {
            v[i] = distr(rnd);
        }
        CUMAT_SAFE_CALL(cudaMemcpyAsync(state_.pointer(), v.data(), 
            numStates * sizeof(state_t), cudaMemcpyHostToDevice, Context::current().stream()));
    }

    /**
     * \brief Creates a new random number generator using the current time as seed.
     */
    SimpleRandom()
        : SimpleRandom(std::chrono::system_clock::now().time_since_epoch().count())
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
    void fillUniform(const MatrixBase<_Derived>& m, const _Scalar& min, const _Scalar& max)
    {
        if (m.size() == 0) return;
        Context& ctx = Context::current();
        KernelLaunchConfig cfg = ctx.createLaunchConfig1D(m.size(), numStates);
        EvaluationKernel <<<1, cfg.thread_per_block.x, 0, ctx.stream() >>>(cfg.virtual_size, m.derived(), min, max, state_.pointer());
        CUMAT_CHECK_ERROR();
    }
};

CUMAT_NAMESPACE_END

#endif