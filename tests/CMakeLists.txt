# Tests

set(CUMAT_TEST_FILES 
  main.cpp
  )

add_executable(cuMat_tests ${CUMAT_TEST_FILES})
target_link_libraries(cuMat_tests Catch)
target_link_libraries(cuMat_tests cuMat)

enable_testing()
add_test(NAME cuMatTest COMMAND cuMat_tests)