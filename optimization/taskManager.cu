#include <map>

#include "taskManager.hpp"

TaskManager *TaskManager::instance = nullptr;

TaskManager *TaskManager::getInstance() {
  if (instance == nullptr) {
    instance = new TaskManager();
  }
  return instance;
}

std::map<GraphNodeId, float> TaskManager::getKernelRunningTimes(cudaGraph_t graph) {
  std::map<GraphNodeId, float> kernelRunningTimes;
  kernelRunningTimes[0] = 1;
  return kernelRunningTimes;
}
