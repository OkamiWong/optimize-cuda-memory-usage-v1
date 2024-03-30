#include <map>

#include "../utilities/logger.hpp"
#include "executor.hpp"
#include "optimization.hpp"
#include "optimizer.hpp"

namespace memopt {

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph) {
  LOG_TRACE();
  return Optimizer::getInstance()->profileAndOptimize(originalGraph);
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

}  // namespace memopt
