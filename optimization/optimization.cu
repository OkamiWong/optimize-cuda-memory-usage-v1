#include <map>

#include "../utilities/logger.hpp"
#include "executor.hpp"
#include "optimization.hpp"
#include "optimizer.hpp"

OptimizationOutput profileAndOptimize(
  cudaGraph_t originalGraph,
  bool optimizeForRepetitiveExecution
) {
  LOG_TRACE();
  return Optimizer::getInstance()->profileAndOptimize(originalGraph, optimizeForRepetitiveExecution);
}

void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraph(
    optimizedGraph,
    executeRandomTask,
    runningTime,
    managedDeviceArrayToHostArrayMap
  );
}

void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
) {
  LOG_TRACE();
  Executor::getInstance()->executeOptimizedGraphRepeatedly(
    optimizedGraph,
    executeRandomTask,
    shouldContinue,
    numIterations,
    runningTime,
    managedDeviceArrayToHostArrayMap
  );
}
