# cuMat: Linear algebra library in CUDA

cuMat strives to be a port of Eigen in CUDA, enabling the performance gain when computing on the GPU.

Overview:
 - Versatile:
   - cuMat supports all matrix and vector sizes, fixed on compile time or dynamically sized during runtime.
   - all matrices can be batched and all operations are parallelized over batches.
   - supports all standard float and integral types, complex types, as well as [custom scalar types](https://shaman42.gitlab.io/cuMat/_advanced__custom_scalar_types.html).
   - supports BLAS 1-3, many reductions, decompositions, and iterative solvers.
   - supports [sparse matrices](https://shaman42.gitlab.io/cuMat/_tutorial_sparse.html).
 - Fast ( [Benchmarks](https://shaman42.gitlab.io/cuMat/_benchmarks.html) ):
   - Kernel merging to minimize memory access.
   - Uses [CUB](https://nvlabs.github.io/cub/) for reductions, [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) for matrix products and [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) for dense decompositions.
   - Custom implementations for all nullary, unary and binary operations.
   - Outperforms cuBLAS if kernel merging can be utilized.
 - Accessible:
   - Simple API influenced by Eigen.
   - Implementation details like context creation and work size spezification are hidden from the user.
   - Thread-safe.
   - Header-only.
   - Cross-Platform support. Developed under Windows, Visual Studio 2017 with CUDA 9.2. Tested with the CI on Linux, gcc and CUDA 9.2.
   - Simple interop to Eigen.

## Motivating example
To demonstrate how cuMat can be used, we show how the code for summing two vectors `a` and `b` into a thrid vector `c` looks like when implemented with Eigen, cuBLAS and cuMat.

**Eigen:**

    Eigen::VectorXf a = ..., b = ...; //some initializations
    Eigen::VectorXf c = a + b; //CPU

**cuBLAS:**

    int n = ...; //size of the vectors
    float* a = ..., b = ...; //some initializations
    float* c = ...; //output memory
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1; //optional scaling factor of b; axpy: c += alpha * b
    cudaMemcpy(c, a, sizeof(float)*n, cudaMemcpyDeviceToDevice); //copy a into c, GPU
    cublasSaxpy(handle, n, &alpha, b, 1, c, 1); //add b to c, GPU
    cublasDestroy(&handle);

Of course, this above code is a bit unfair because the boilerplate code of creating the cuBLAS handle is included.
In practice, this has to be done only once, so the above code reduces to two lines, the memcpy and the axpy.

**cuMat:**

    cuMat::VectorXf a = ..., b = ...; //some initialization
    cuMat::VectorXf c = a + b; //GPU

## Documentation
The documentation can be found under [https://shaman42.gitlab.io/cuMat/](https://shaman42.gitlab.io/cuMat/_getting_started.html).
All other open questions regarding this library are answered there.

## Requirements
cuMat is header-only, but it builds on some third-party libraries:
 - cuBLAS, cuSOLVER: shipped with the CUDA SDK.
 - [CUB](https://nvlabs.github.io/cub/): can be found inside Thrust as part of the CUDA SDK, in the third-party folder of cuMat, or provide your own version.
 - (Optional) [Eigen](http://eigen.tuxfamily.org) for printing matrices and for the Eigen interop. A working version can be found in the third-party folder.

## License
cuMat is shipped under the permissive [MIT](https://choosealicense.com/licenses/mit/) license.

## Bug reports
If you find bugs in the library, feel free to open an issue. I will continue to use this library in future projects and therefore continue to improve and extend this library. Of course, pull requests are more than welcome.