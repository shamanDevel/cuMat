# Batched Reductions

set(CUMAT_BENCHMARK_BATCHED_REDUCTIONS
  ../json_st.h
  ../json_st.cpp
  ../Json.h
  ../Json.cpp
  main.cpp
  reductions.cuh
  reductions.h
  reductions.cu
  MakePlotsLinear.py
  MakePlotsFull.py
  AlgorithmDecision.py
  configuration.json
  )
  
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
endif()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)

cuda_add_executable(
	batched_reductions 
	${CUMAT_BENCHMARK_BATCHED_REDUCTIONS})
cuda_add_cublas_to_target(batched_reductions)
set_target_properties(batched_reductions PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(batched_reductions PROPERTIES FOLDER Benchmarks)
target_link_libraries(batched_reductions ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
target_compile_definitions(batched_reductions PRIVATE 
	CUMAT_EIGEN_SUPPORT=1 
	CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/configuration.json
	PYTHON_FILES=${CMAKE_CURRENT_SOURCE_DIR}/
	OUTPUT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../
)
