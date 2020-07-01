# Header-only library

# Define 'cuMat_headers' variable to list all the header files
# Todo: maybe replace by GLOB
set(CUMAT_HEADERS
  src/Macros.h 
  src/ForwardDeclarations.h
  src/Profiling.h
  src/Logging.h
  src/Errors.h
  src/Constants.h
  src/Context.h
  src/Allocator.h
  src/NumTraits.h
  src/DevicePointer.h
  src/EigenInteropHelpers.h
  src/MatrixBase.h
  src/Matrix.h
  src/IO.h
  src/CwiseOp.h
  src/MatrixBlock.h
  src/MatrixBlockPluginLvalue.inl
  src/MatrixBlockPluginRvalue.inl
  src/NullaryOps.h
  src/MatrixNullaryOpsPlugin.inl
  src/UnaryOps.h
  src/UnaryOpsPlugin.inl
  src/CudaUtils.h
  src/TransposeOp.h
  src/BinaryOps.h
  src/BinaryOpsPlugin.inl
  src/ReductionOps.h
  src/ReductionOpsPlugin.inl
  src/ReductionAlgorithmSelection.h
  src/Iterator.h
  src/CublasApi.h
  src/SimpleRandom.h
  src/ProductOp.h
  Core
  
  src/CusolverApi.h
  src/SolverBase.h
  src/DecompositionBase.h
  src/LUDecomposition.h
  src/CholeskyDecomposition.h
  src/DenseLinAlgOps.h
  src/DenseLinAlgPlugin.inl
  Dense
  
  src/SparseMatrixBase.h
  src/SparseMatrix.h
  src/SparseEvaluation.h
  src/SparseProductEvaluation.h
  src/SparseExpressionOp.h
  src/SparseExpressionOpPlugin.inl
  Sparse
  
  src/IterativeSolverBase.h
  src/ConjugateGradient.h
  IterativeLinearSolvers
  )

add_library(cuMat INTERFACE) # 'moduleA' is an INTERFACE pseudo target

#
# From here, the target 'moduleA' can be customised
#
target_include_directories(cuMat INTERFACE ${CMAKE_SOURCE_DIR}) # Transitively forwarded
target_include_directories(cuMat INTERFACE ${CUDA_INCLUDE_DIRS})
#install(TARGETS cuMat ...)

#
#  HACK: have the files showing in the IDE, under the name 'moduleA_ide'
#
option(CUMAT_SOURCE_LIBRARY "show the source headers as a library in the IDE" OFF)
if(CUMAT_SOURCE_LIBRARY)
cuda_add_library(cuMat_ide ${CUMAT_HEADERS})
target_include_directories(cuMat_ide INTERFACE ${CMAKE_SOURCE_DIR}) # Transitively forwarded
target_include_directories(cuMat_ide INTERFACE ${CUDA_INCLUDE_DIRS})
endif(CUMAT_SOURCE_LIBRARY)
