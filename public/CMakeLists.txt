add_library(memopt STATIC memopt.cu)
target_link_libraries(
  memopt
  STATIC
  utilities
  cudaUtilities
  annotation
  optimization
  peakMemoryUsageProfiler
)
