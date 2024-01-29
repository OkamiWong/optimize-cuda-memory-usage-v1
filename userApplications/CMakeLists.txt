add_executable(tiledCholesky tiledCholesky.cu)
target_link_libraries(
  tiledCholesky
  utilities
  cudaUtilities
  annotation
  optimization
  peakMemoryUsageProfiler
  logger
  CUDA::cublas
  CUDA::cusolver
  fmt::fmt
)
