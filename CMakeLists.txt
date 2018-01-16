cmake_minimum_required(VERSION 3.0.0)
project(cuMat)
# Main CMakeLists of cuMat

# Cuda is always required
find_package(CUDA REQUIRED)

# Eigen is not required by default,
# but the tests depend on it -> use bundled Eigen for tests

# Hack: The FindCUDA that is used by the CI does not correctly support target_include_directories,
#  it does not forward the include paths.
#  Hence I have to include the paths globally.
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third-party)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(third-party)
add_subdirectory(cuMat)
add_subdirectory(tests)
add_subdirectory(demos)
add_subdirectory(docs)
add_subdirectory(benchmarks)

