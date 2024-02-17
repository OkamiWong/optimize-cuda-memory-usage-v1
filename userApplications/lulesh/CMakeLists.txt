add_executable(lulesh lulesh.cu)
target_link_libraries(
  lulesh
  utilities
  cudaUtilities
  annotation
  memoryManager
  optimization
  peakMemoryUsageProfiler
  logger
  fmt::fmt
)
