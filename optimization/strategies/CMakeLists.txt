add_library(strategyUtilities SHARED strategyUtilities.cu)
target_link_libraries(
  strategyUtilities
  logger
  fmt::fmt
)

add_library(strategies SHARED noOptimizationStrategy.cu twoStepOptimizationStrategy.cu)
target_link_libraries(
  strategies
  strategyUtilities
  firstStepSolver
  secondStepSolver
  logger
)

add_library(firstStepSolver SHARED firstStepSolver.cpp)
target_link_libraries(
  firstStepSolver
  logger
  fmt::fmt
)

add_library(secondStepSolver SHARED secondStepSolver.cpp)
target_link_libraries(
  secondStepSolver
  logger
  fmt::fmt
  ortools::ortools
)
