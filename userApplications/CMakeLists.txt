add_executable(artificialApplication artificialApplication.cu)
target_link_libraries(
  artificialApplication
  utilities
  cudaUtilities
  annotation
  optimization
  memoryManager
  logger
)

add_executable(tiledCholesky tiledCholesky.cu)
target_link_libraries(
  tiledCholesky
  utilities
  cudaUtilities
  annotation
  optimization
  memoryManager
  logger
  ${CUDA_cublas_LIBRARY}
  ${CUDA_cusolver_LIBRARY}
  fmt::fmt
)
