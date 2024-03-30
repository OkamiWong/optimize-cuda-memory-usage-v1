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
target_include_directories(
  memopt
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
