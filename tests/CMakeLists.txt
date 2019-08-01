# Tests

option(CUMAT_SEPARATE_TESTS "compile each test into a separate project / executable" OFF)

set(CMAKE_BUILD_TYPE Debug)

set(CUMAT_TEST_HEADERS
  Barrier.h
  Utils.h
  TestUnaryOps.cuh
  TestBinaryOps.cuh
  )

set(CUMAT_TEST_FILES 
  TestContext.cu
  TestDevicePointer.cu
  TestMatrix.cu
  TestEigenInterop.cu
  TestNullaryOps.cu
  TestMatrixBlock.cu
  TestUnaryOps1.cu
  TestUnaryOps2.cu
  TestUnaryOps3.cu
  TestUnaryOps4.cu
  TestCasting.cu
  TestDiagonal.cu
  TestTranspose.cu
  TestSwapAxisOp.cu
  TestBinaryOps1.cu
  TestBinaryOps2.cu
  TestBinaryOps3.cu
  TestReductionOps1a.cu
  TestReductionOps1b.cu
  TestReductionOps2.cu
  TestReductionOps3.cu
  TestReductionOps4.cu
  TestReductionOps5.cu
  TestIterator.cu
  TestRandom.cu
  TestMatrixMultOp.cu
  TestComplexOps.cu
  TestLUDecomposition.cu
  TestCholeskyDecomposition.cu
  TestLinAlgOps1.cu
  TestLinAlgOps2.cu
  TestLinAlgOps3.cu
  TestLinAlgOps4.cu
  TestCompoundAssignment.cu
  TestSparseMatrix.cu
  TestConjugateGradient.cu
  TestSparseMultOp.cu
  TestBlockedConjugateGradient.cu
  
  BenchmarkDenseConjugateGradient.cu
  )

if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --device-debug)
endif()
enable_testing()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DCUMAT_CONTEXT_DEBUG_MEMORY=1")
#add_definitions(-DCUMAT_EIGEN_SUPPORT=1)

if(CUMAT_SEPARATE_TESTS)
    # a separate project for each test file
    foreach( testsourcefile ${CUMAT_TEST_FILES} )
        string( REPLACE ".cpp" "" testname ${testsourcefile} )
        cuda_add_executable(${testname} ${CUMAT_TEST_HEADERS} ${testsourcefile} main.cpp)
        cuda_add_cublas_to_target(${testname})
        set_target_properties(${testname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(${testname} PROPERTIES FOLDER Tests)
        
        # Somehow, the CI does not understand the include paths if using target_link_libraries
        #  -> I have to set the globally in the root folder
        #target_link_libraries(${testname} Eigen3)
        #target_link_libraries(${testname} Catch)
        #target_link_libraries(${testname} cuMat)
        
        target_link_libraries(${testname} ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
        #target_compile_definitions(${testname} PUBLIC CUMAT_CONTEXT_DEBUG_MEMORY=1 CUMAT_EIGEN_SUPPORT=1 CUMAT_PROFILING=1)
    endforeach(testsourcefile)
else(CUMAT_SEPARATE_TESTS)
    # only one project
    cuda_add_executable(cuMat_tests ${CUMAT_TEST_HEADERS} ${CUMAT_TEST_FILES} main.cpp)
    cuda_add_cublas_to_target(cuMat_tests)
    set_target_properties(cuMat_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    # Somehow, the CI does not understand the include paths if using target_link_libraries
    #  -> I have to set the globally in the root folder
    #target_link_libraries(cuMat_tests Eigen3)
    #target_link_libraries(cuMat_tests Catch)
    #target_link_libraries(cuMat_tests cuMat)
    
    target_link_libraries(cuMat_tests ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
    #target_compile_definitions(cuMat_tests PUBLIC CUMAT_CONTEXT_DEBUG_MEMORY=1 CUMAT_EIGEN_SUPPORT=1 CUMAT_PROFILING=1)
    add_test(NAME cuMatTest COMMAND cuMat_tests)
endif(CUMAT_SEPARATE_TESTS)

# Special test cases

cuda_add_executable(TestNoCUDA Utils.h TestNoCUDA.h TestNoCUDA.cu TestNoCUDA.cpp main.cpp)
cuda_add_cublas_to_target(TestNoCUDA)
set_target_properties(TestNoCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(TestNoCUDA PROPERTIES FOLDER Tests)
