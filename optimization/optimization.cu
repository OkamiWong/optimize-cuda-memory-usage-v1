#include "optimization.hpp"
#include "optimizer.hpp"
#include "taskManager.hpp"

CustomGraph profileAndOptimize(cudaGraph_t originalGraph) {
  return Optimizer::getInstance()->profileAndOptimize(originalGraph);
}

void executeOptimizedGraph(const CustomGraph& optimizedGraph) {
  TaskManager::getInstance()->executeOptimizedGraph(optimizedGraph);
}
