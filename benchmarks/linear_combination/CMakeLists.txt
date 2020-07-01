# Linear Combinations

set(CUMAT_BENCHMARK_LINEAR_COMBINATIONS
  ../json_st.h
  ../json_st.cpp
  ../Json.h
  ../Json.cpp
  main.cpp
  benchmark.h
  Implementation_cuBlas.cu
  Implementation_cuMat.cu
  Implementation_Eigen.cpp
  Implementation_numpy.py
  Implementation_tensorflow.py
  MakePlots.py
  configuration.json
  )
  
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
endif()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)

cuda_add_executable(
	linear_combination 
	${CUMAT_BENCHMARK_LINEAR_COMBINATIONS})
cuda_add_cublas_to_target(linear_combination)
set_target_properties(linear_combination PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(linear_combination PROPERTIES FOLDER Benchmarks)
target_link_libraries(linear_combination ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
target_compile_definitions(linear_combination PRIVATE 
	CUMAT_EIGEN_SUPPORT=1 
	CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/configuration.json
	PYTHON_FILES=${CMAKE_CURRENT_SOURCE_DIR}/
	OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(linear_combination OpenMP::OpenMP_CXX)
endif()

# MKL
find_package(MKL QUIET)
if(MKL_FOUND)
	target_include_directories(linear_combination PRIVATE ${MKL_INCLUDE_DIRS})
	target_link_libraries(linear_combination ${MKL_LIBRARIES})
	target_compile_definitions(linear_combination PRIVATE EIGEN_USE_MKL_ALL)
	message("Compile Eigen benchmarks with MKL support")
else()
	message("MKL not found. If you are missing FindMKL.cmake, you can download it at https://gist.github.com/scivision/5108cf6ab1515f581a84cd9ad1ef72aa")
endif()