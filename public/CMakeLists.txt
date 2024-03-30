add_library(memopt INTERFACE)
target_link_libraries(
  memopt
  INTERFACE
  utilities
  cudaUtilities
  annotation
  optimization
  peakMemoryUsageProfiler
)
