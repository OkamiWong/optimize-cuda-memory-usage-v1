add_library(strategyUtilities SHARED strategyUtilities.cu)
target_link_libraries(
  strategyUtilities
  fmt::fmt
)

add_library(strategies SHARED noOptimizationStrategy.cu twoStepOptimizationStrategy.cu)
target_link_libraries(
  strategies
  strategyUtilities
  integerProgrammingSolver
  logger
)

add_library(integerProgrammingSolver SHARED integerProgrammingSolver.cpp)
target_link_libraries(
  integerProgrammingSolver
  ortools::ortools
  logger
)
