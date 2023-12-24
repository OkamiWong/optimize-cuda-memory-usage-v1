add_library(strategyUtilities SHARED strategyUtilities.cu)
target_link_libraries(
  strategyUtilities
  fmt::fmt
)

add_library(strategies SHARED twoStepOptimizationStrategy.cu)
target_link_libraries(
  strategies
  strategyUtilities
  logger
)
