#include <type_traits>

#include "optimizer.hpp"
#include "strategies/strategies.hpp"
#include "taskManager.hpp"

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

CustomGraph Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  // Profile
  auto taskManager = TaskManager::getInstance();
  taskManager->registerDummyKernelHandle(originalGraph);
  auto cuGraphNodeToKernelDurationMap = taskManager->getCuGraphNodeToKernelDurationMap(originalGraph);

  // Optimize
  auto customGraph = this->optimize<PrefetchOnlyStrategy>(originalGraph, cuGraphNodeToKernelDurationMap);

  return customGraph;
}