#include <map>

#include "../utilities/logger.hpp"
#include "executor.hpp"
#include "optimization.hpp"
#include "optimizer.hpp"

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph) {
  LOG_TRACE();
  return Optimizer::getInstance()->profileAndOptimize(originalGraph);
}

float executeOptimizedGraph(OptimizationOutput& optimizedGraph, ExecuteRandomTask executeRandomTask) {
  LOG_TRACE();
  return Executor::getInstance()->executeOptimizedGraph(optimizedGraph, executeRandomTask);
}
