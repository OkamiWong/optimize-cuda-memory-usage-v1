add_library(
  optimization
  SHARED
  optimization.cu
  taskManager.cu
  optimizer.cu
)
target_link_libraries(
  optimization
  strategies
  annotation
  cudaUtilities
  cudaGraphUtilities
  cudaGraphExecutionTimelineProfiler
  logger
  -lcuda
)

add_subdirectory(strategies)