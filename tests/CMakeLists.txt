# Tests

set(CUMAT_TEST_FILES 
  main.cpp
  TestContext.cpp
  Barrier.h
  TestDevicePointer.cpp
  )

add_executable(cuMat_tests ${CUMAT_TEST_FILES})
set_target_properties(cuMat_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuMat_tests Catch)
target_link_libraries(cuMat_tests cuMat)
target_link_libraries(cuMat_tests ${CUDA_LIBRARIES})
target_compile_definitions(cuMat_tests PRIVATE CUMAT_CONTEXT_DEBUG_MEMORY=1)

enable_testing()
add_test(NAME cuMatTest COMMAND cuMat_tests)