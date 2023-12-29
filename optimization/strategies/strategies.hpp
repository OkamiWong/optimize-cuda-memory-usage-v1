#pragma once

#include "../optimizer.hpp"

class NoOptimizationStrategy {
  public:
    CustomGraph run(OptimizationInput &optimizationInput);
};

class TwoStepOptimizationStrategy {
  public:
    CustomGraph run(OptimizationInput &optimizationInput);
};
