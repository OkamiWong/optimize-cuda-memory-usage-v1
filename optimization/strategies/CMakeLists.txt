add_library(strategyUtilities SHARED strategyUtilities.cu)
target_link_libraries(
  strategyUtilities
  fmt::fmt
)

add_library(strategies SHARED noOptimizationStrategy.cu twoStepOptimizationStrategy.cu)
target_link_libraries(
  strategies
  strategyUtilities
  firstStepSolver
  secondStepSolver
)

add_library(firstStepSolver SHARED firstStepSolver.cpp)
target_link_libraries(
  firstStepSolver
  fmt::fmt
)

add_library(secondStepSolver SHARED secondStepSolver.cpp)
target_link_libraries(
  secondStepSolver
  utilities
  fmt::fmt
  ortools::ortools
)
