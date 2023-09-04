#include "../utilities/logger.hpp"
#include "optimization.hpp"
#include "optimizer.hpp"
#include "taskManager.hpp"

CustomGraph profileAndOptimize(cudaGraph_t originalGraph) {
  LOG_TRACE();
  return Optimizer::getInstance()->profileAndOptimize(originalGraph);
}

void executeOptimizedGraph(CustomGraph& optimizedGraph) {
  LOG_TRACE();
  TaskManager::getInstance()->executeOptimizedGraph(optimizedGraph);
}
