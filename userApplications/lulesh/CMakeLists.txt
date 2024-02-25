add_executable(lulesh lulesh.cu)
target_link_libraries(
  lulesh
  utilities
  cudaUtilities
  annotation
  optimization
  peakMemoryUsageProfiler
  fmt::fmt
)
