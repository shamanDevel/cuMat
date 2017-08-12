# Header-only library

# Define 'cuMat_headers' variable to list all the header files
# Todo: maybe replace by GLOB
set(CUMAT_HEADERS 
  Matrix.h
  )

add_library(cuMat INTERFACE) # 'moduleA' is an INTERFACE pseudo target

#
# From here, the target 'moduleA' can be customised
#
target_include_directories(cuMat INTERFACE ../) # Transitively forwarded
#install(TARGETS cuMat ...)

#
#  HACK: have the files showing in the IDE, under the name 'moduleA_ide'
#
add_custom_target(cuMat_ide SOURCES ${CUMAT_HEADERS})