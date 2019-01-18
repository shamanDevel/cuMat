# SparseMatrix-DenseVector product (CSRMV)

set(CUMAT_BENCHMARK_CSRMV
  ../json_st.h
  ../json_st.cpp
  ../Json.h
  ../Json.cpp
  main.cpp
  benchmark.h
  Implementation_cuBlas.cu
  Implementation_cuMat_CSR.cu
  Implementation_cuMat_ELLPACK.cu
  Implementation_Eigen.cpp
  MakePlots.py
  configuration.json
  )
  
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
endif()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)

cuda_add_executable(
	csrmv 
	${CUMAT_BENCHMARK_CSRMV})
cuda_add_cublas_to_target(csrmv)
set_target_properties(csrmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(csrmv PROPERTIES FOLDER Benchmarks)
target_link_libraries(csrmv ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})
target_compile_definitions(csrmv PRIVATE 
	CUMAT_EIGEN_SUPPORT=1 
	CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/configuration.json
	PYTHON_FILES=${CMAKE_CURRENT_SOURCE_DIR}/
	OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(csrmv OpenMP::OpenMP_CXX)
endif()

# MKL
find_package(MKL)
if(MKL_FOUND)
	target_include_directories(csrmv PRIVATE ${MKL_INCLUDE_DIRS})
	target_link_libraries(csrmv ${MKL_LIBRARIES})
	target_compile_definitions(csrmv PRIVATE EIGEN_USE_MKL_ALL)
endif()