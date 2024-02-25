add_executable(tiledCholesky tiledCholesky.cu)
target_link_libraries(
  tiledCholesky
  utilities
  cudaUtilities
  annotation
  optimization
  peakMemoryUsageProfiler
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)

add_subdirectory(lulesh)
