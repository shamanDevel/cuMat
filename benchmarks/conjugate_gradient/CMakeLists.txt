# Conjugate Gradient

set(CUMAT_BENCHMARK_CONJUGATE_GRADIENT
  ../json_st.h
  ../json_st.cpp
  ../Json.h
  ../Json.cpp
  main.cpp
  benchmark.h
  Implementation_cuMat.cu
  Implementation_Eigen.cpp
  MakePlots.py
  configuration.json
  )
  
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
endif()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)

cuda_add_executable(
	conjugate_gradient 
	${CUMAT_BENCHMARK_CONJUGATE_GRADIENT})
cuda_add_cublas_to_target(conjugate_gradient)
set_target_properties(conjugate_gradient PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(conjugate_gradient PROPERTIES FOLDER Benchmarks)
target_link_libraries(conjugate_gradient ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})
target_compile_definitions(conjugate_gradient PRIVATE 
	CUMAT_EIGEN_SUPPORT=1 
	CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/configuration.json
	PYTHON_FILES=${CMAKE_CURRENT_SOURCE_DIR}/
	OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(conjugate_gradient OpenMP::OpenMP_CXX)
endif()

# MKL
find_package(MKL)
if(MKL_FOUND)
	target_include_directories(conjugate_gradient PRIVATE ${MKL_INCLUDE_DIRS})
	target_link_libraries(conjugate_gradient ${MKL_LIBRARIES})
	target_compile_definitions(conjugate_gradient PRIVATE EIGEN_USE_MKL_ALL)
endif()