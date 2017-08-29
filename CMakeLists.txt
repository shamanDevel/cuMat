cmake_minimum_required(VERSION 3.0.0)
project(cuMat)
# Main CMakeLists of cuMat

# Cuda is always required
find_package(CUDA REQUIRED)

# Check if Eigen is supported
find_package (Eigen3)
option(CUMAT_EIGEN_SUPPORT "support Eigen interop (if Eigen is found)" TRUE)
if(EIGEN3_FOUND)
  if(CUMAT_EIGEN_SUPPORT)
    # Enable Eigen interop
	message(STATUS "Eigen interop enabled")
	add_definitions(-DCUMAT_EIGEN_SUPPORT=1)
	include_directories(${EIGEN3_INCLUDE_DIRS})
  else()
    message(STATUS "Eigen found, but interop is disabled")
  endif()
else()
  message(STATUS "Eigen not found")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

add_subdirectory(third-party)
add_subdirectory(cuMat)
add_subdirectory(tests)
add_subdirectory(docs)