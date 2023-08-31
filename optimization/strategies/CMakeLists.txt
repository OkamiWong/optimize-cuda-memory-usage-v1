add_library(strategyUtilities SHARED strategyUtilities.cu)
target_link_libraries(strategyUtilities -lcuda)

add_library(strategies SHARED prefetchOnlyStrategy.cu)
target_link_libraries(
  strategies
  strategyUtilities
  cudaGraphUtilities
)
