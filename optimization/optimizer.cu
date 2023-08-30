#include "optimizer.hpp"
#include "taskManager.hpp"

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

CustomGraph Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  auto taskManager = TaskManager::getInstance();

  taskManager->registerDummyKernelHandle(originalGraph);

  auto kernelRunningTimes = taskManager->getKernelRunningTimes(originalGraph);

  CustomGraph temp;
  return temp;
}
