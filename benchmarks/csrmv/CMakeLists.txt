# SparseMatrix-DenseVector product (CSRMV)

set(CUMAT_BENCHMARK_CSRMV
  ../json_st.h
  ../json_st.cpp
  ../Json.h
  ../Json.cpp
  main.cpp
  benchmark.h
  Implementation_cuBlas.cu
  Implementation_cuMat.cu
  Implementation_Eigen.cpp
  MakePlots.py
  )
  
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
endif()

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