# Header-only library

# Define 'cuMat_headers' variable to list all the header files
# Todo: maybe replace by GLOB
set(CUMAT_HEADERS
  src/Macros.h 
  src/ForwardDeclarations.h
  src/Logging.h
  src/Errors.h
  src/Constants.h
  src/Context.h
  src/Allocator.h
  src/DevicePointer.h
  src/EigenInteropHelpers.h
  src/MatrixBase.h
  src/Matrix.h
  src/CwiseOp.h
  src/NullaryOps.h
  src/MatrixBlock.h

  Core
  )

add_library(cuMat INTERFACE) # 'moduleA' is an INTERFACE pseudo target

#
# From here, the target 'moduleA' can be customised
#
target_include_directories(cuMat INTERFACE ../) # Transitively forwarded
target_include_directories(cuMat INTERFACE ${CUDA_INCLUDE_DIRS})
#install(TARGETS cuMat ...)

#
#  HACK: have the files showing in the IDE, under the name 'moduleA_ide'
#
cuda_add_library(cuMat_ide ${CUMAT_HEADERS})
target_include_directories(cuMat_ide INTERFACE ../) # Transitively forwarded
target_include_directories(cuMat_ide INTERFACE ${CUDA_INCLUDE_DIRS})