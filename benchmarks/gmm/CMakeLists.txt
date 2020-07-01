# Linear Combinations

set(CUMAT_BENCHMARK_GMM
  ../json_st.h
  ../json_st.cpp
  ../Json.h
  ../Json.cpp
  main.cpp
  benchmark.h
  Implementation_cuMat.cu
  Implementation_Eigen.cpp
  Implementation_tensorflow.py
  MakePlots1.py
  MakePlots2.py
  GenerateData.py
  configuration.json
  )
  
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
endif()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)

cuda_add_executable(
	gmm 
	${CUMAT_BENCHMARK_GMM})
cuda_add_cublas_to_target(gmm)
set_target_properties(gmm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gmm PROPERTIES FOLDER Benchmarks)
target_link_libraries(gmm ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
target_compile_definitions(gmm PRIVATE 
	CUMAT_EIGEN_SUPPORT=1 
	CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/configuration.json
	PYTHON_FILES=${CMAKE_CURRENT_SOURCE_DIR}/
	OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../
)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gmm OpenMP::OpenMP_CXX)
endif()

# MKL
find_package(MKL QUIET)
if(MKL_FOUND)
	target_include_directories(gmm PRIVATE ${MKL_INCLUDE_DIRS})
	target_link_libraries(gmm ${MKL_LIBRARIES})
	target_compile_definitions(gmm PRIVATE EIGEN_USE_MKL_ALL)
    message("Compile Eigen benchmarks with MKL support")
endif()