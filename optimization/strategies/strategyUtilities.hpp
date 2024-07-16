#pragma once

#include "../optimizationInput.hpp"
#include "../optimizationOutput.hpp"

namespace memopt {

void printOptimizationInput(OptimizationInput &input);
void printOptimizationOutput(OptimizationOutput &output, int stageIndex);

}  // namespace memopt
