#pragma once

#include "../optimizer.hpp"

namespace memopt {

class NoOptimizationStrategy {
  public:
    OptimizationOutput run(OptimizationInput &optimizationInput);
};

class TwoStepOptimizationStrategy {
  public:
    OptimizationOutput run(OptimizationInput &optimizationInput);
};

}  // namespace memopt
