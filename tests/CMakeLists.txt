# Tests

set(CUMAT_TEST_FILES 
  main.cpp
  
  Barrier.h
  Utils.h

  TestContext.cu
  TestDevicePointer.cu
  TestMatrix.cu
  TestEigenInterop.cu
  TestNullaryOps.cu
  TestMatrixBlock.cu
  TestUnaryOps.cuh
  TestUnaryOps1.cu
  TestUnaryOps2.cu
  TestUnaryOps3.cu
  TestUnaryOps4.cu
  TestCasting.cu
  TestTranspose.cu
  TestBinaryOps.cuh
  TestBinaryOps1.cu
  )

list(APPEND CUDA_NVCC_FLAGS --cl-version=2017 --device-debug -Xcompiler /bigobj)
cuda_add_executable(cuMat_tests ${CUMAT_TEST_FILES})
cuda_add_cublas_to_target(cuMat_tests)
set_target_properties(cuMat_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuMat_tests Catch)
target_link_libraries(cuMat_tests cuMat)
target_link_libraries(cuMat_tests ${CUDA_LIBRARIES})
target_compile_definitions(cuMat_tests PRIVATE CUMAT_CONTEXT_DEBUG_MEMORY=1 CUMAT_PROFILING=1)

enable_testing()
add_test(NAME cuMatTest COMMAND cuMat_tests)