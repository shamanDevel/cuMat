cmake_minimum_required(VERSION 3.0.0)
project(cuMat)
# Main CMakeLists of cuMat

# Cuda is always required
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third-party)
add_subdirectory(cuMat)
add_subdirectory(tests)
