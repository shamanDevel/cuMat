# Third party libraries

# Adds a target 'Catch' containing the catch file

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ./)

add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE ./)