# Benchmarks

set(CUMAT_BENCHMARK_COMMON
  json_st.h
  json_st.cpp
  Json.h
  Json.cpp
  main.cpp
  benchmark.h
  )

set(CUMAT_BENCHMARKS
  linear_combination
  )

list(APPEND CUDA_NVCC_FLAGS --cl-version=2017)
enable_testing()

# create a separate project for each benchmark
foreach( benchmark ${CUMAT_BENCHMARKS} )
    cuda_add_executable(
        ${benchmark} 
        ${CUMAT_BENCHMARK_COMMON} 
        ${benchmark}/Implementation_Eigen.cpp
        ${benchmark}/Implementation_cuMat.cu
        ${benchmark}/Implementation_numpy.py
        ${benchmark}/Implementation_tensorflow.py)
    cuda_add_cublas_to_target(${benchmark})
    set_target_properties(${benchmark} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${benchmark} PROPERTIES FOLDER Benchmarks)
    target_link_libraries(${benchmark} ${CUDA_LIBRARIES})
    target_compile_definitions(${benchmark} PRIVATE 
        CUMAT_EIGEN_SUPPORT=1 
        CUMAT_PROFILING=1 
        CONFIG_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}/configuration.json
        PYTHON_FILES=${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}/
    )
endforeach(benchmark)
