#include "strategies.hpp"
#include "strategyUtilities.hpp"
#include "../../utilities/logger.hpp"

CustomGraph TwoStepOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  printOptimizationInput(input);

  CustomGraph optimizedGraph;
  return optimizedGraph;
}
