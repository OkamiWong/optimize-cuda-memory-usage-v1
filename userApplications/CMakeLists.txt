add_executable(tiledCholesky tiledCholesky.cu)
target_link_libraries(
  tiledCholesky
  utilities
  cudaUtilities
  annotation
  memoryManager
  optimization
  peakMemoryUsageProfiler
  logger
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)
