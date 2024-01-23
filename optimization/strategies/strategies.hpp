#pragma once

#include "../optimizer.hpp"

class NoOptimizationStrategy {
  public:
    OptimizationOutput run(OptimizationInput &optimizationInput);
};

class TwoStepOptimizationStrategy {
  public:
    OptimizationOutput run(OptimizationInput &optimizationInput);
};
