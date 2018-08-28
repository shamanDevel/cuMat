#ifndef __CUMAT_PROFILING_H__
#define __CUMAT_PROFILING_H__

#include "Macros.h"

#ifndef CUMAT_PROFILING
/**
 * \brief Define this macro as '1' to enable profiling.
 * For the created statistics, see the class Profiling.
 */
#define CUMAT_PROFILING 0
#endif

CUMAT_NAMESPACE_BEGIN



/**
 * \brief This class contains the counters used to profile the library.
 * See Counter for which statistics are available.
 * For easy access, several macros are available with the prefix CUMAT_PROFILING_ .
 */
class Profiling
{
public:
    enum Counter
    {
        DeviceMemAlloc,
        DeviceMemFree,
        HostMemAlloc,
        HostMemFree,

        MemcpyDeviceToDevice,
        MemcpyHostToHost,
        MemcpyDeviceToHost,
        MemcpyHostToDevice,

        /**
         * \brief Any evaluation has happend
         */
        EvalAny,
        /**
         * \brief Component-wise evaluation
         */
        EvalCwise,
        /**
         * \brief Special transposition operation
         */
        EvalTranspose,
        /**
         * \brief Reduction operation with CUB
         */
        EvalReduction,
        /**
         * \brief Matrix-Matrix multiplication with cuBLAS
         */
        EvalMatmul,
        /**
         * \brief Sparse component-wise evaluation
         */
        EvalCwiseSparse,
        /**
         * \brief Sparse matrix-matrix multiplication
         */
        EvalMatmulSparse,

        _NumCounter_
    };
private:
    size_t counters_[_NumCounter_];

public:
    void resetAll()
    {
        for (size_t i = 0; i < _NumCounter_; ++i) counters_[i] = 0;
    }
    void reset(Counter counter)
    {
        counters_[counter] = 0;
    }
    void inc(Counter counter)
    {
        counters_[counter]++;
    }
    size_t get(Counter counter)
    {
        return counters_[counter];
    }
    size_t getReset(Counter counter)
    {
        size_t v = counters_[counter];
        counters_[counter] = 0;
        return v;
    }

private:
    Profiling()
    {
        resetAll();
    }
    CUMAT_DISALLOW_COPY_AND_ASSIGN(Profiling);

public:
    static Profiling& instance()
    {
        static Profiling p;
        return p;
    }
};

#ifdef CUMAT_PARSED_BY_DOXYGEN

/**
 * \brief Increments the counter 'counter' (an element of Profiling::Counter).
 */
#define CUMAT_PROFILING_INC(counter)

/**
 * \brief Gets and resets the counter 'counter' (an element of Profiling::Counter).
 * If profiling is disabled (CUMAT_PROFILING is not defined or unequal to 1), the result
 *  is not defined.
 */
#define CUMAT_PROFILING_GET(counter)

/**
 * \brief Resets all counters
 */
#define CUMAT_PROFILING_RESET()

#else

#if CUMAT_PROFILING==1
//Profiling enabled
#define CUMAT_PROFILING_INC(counter) \
    CUMAT_NAMESPACE Profiling::instance().inc(CUMAT_NAMESPACE Profiling::Counter::counter)
#define CUMAT_PROFILING_GET(counter) \
    CUMAT_NAMESPACE Profiling::instance().getReset(CUMAT_NAMESPACE Profiling::Counter::counter)
#define CUMAT_PROFILING_RESET() \
    CUMAT_NAMESPACE Profiling::instance().resetAll()
#else
//Profiling disabled
#define CUMAT_PROFILING_INC(counter) ((void)0)
#define CUMAT_PROFILING_GET(counter) ((void)0)
#define CUMAT_PROFILING_RESET() ((void)0)
#endif

#endif



CUMAT_NAMESPACE_END


#endif