#pragma once

#include "../optimizer.hpp"

class TwoStepOptimizationStrategy {
  public:
    CustomGraph run(OptimizationInput optimizationInput);
};
