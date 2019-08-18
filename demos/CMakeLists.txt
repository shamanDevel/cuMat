# Demos

set(CUMAT_DEMO_FILES 
  Transposing.cu
  Test.cu
  BuddhabrotSimple.cu
  BuddhabrotComplex.cu
  )

if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
list(APPEND CUDA_NVCC_FLAGS --device-debug)
endif()

add_definitions(-DCUMAT_EIGEN_SUPPORT=1 -DCUMAT_PROFILING=1)

# a separate project for each demo file
foreach( demosourcefile ${CUMAT_DEMO_FILES} )
    string( REPLACE ".cu" "" demoname ${demosourcefile} )
    cuda_add_executable(${demoname} ${demosourcefile})
    cuda_add_cublas_to_target(${demoname})
    set_target_properties(${demoname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${demoname} PROPERTIES FOLDER Demos)
    
    # Somehow, the CI does not understand the include paths if using target_link_libraries
    #  -> I have to set the globally in the root folder
    #target_link_libraries(${demoname} Eigen3)
    #target_link_libraries(${demoname} Catch)
    #target_link_libraries(${demoname} cuMat)
    
    target_link_libraries(${demoname} ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY})
    #target_compile_definitions(${demoname} PRIVATE CUMAT_EIGEN_SUPPORT=1 CUMAT_PROFILING=1)
endforeach(demosourcefile)
