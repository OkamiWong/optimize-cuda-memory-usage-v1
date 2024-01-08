#include <map>

#include "../profiling/memoryManager.hpp"
#include "../utilities/constants.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "optimization.hpp"
#include "optimizer.hpp"
#include "taskManager.hpp"

CustomGraph profileAndOptimize(cudaGraph_t originalGraph) {
  LOG_TRACE();
  return Optimizer::getInstance()->profileAndOptimize(originalGraph);
}

void distributeInitialData(CustomGraph& optimizedGraph) {
  LOG_TRACE();

  std::map<void*, bool> visited;
  for (const auto& [ptr, size] : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    visited[ptr] = true;
    checkCudaErrors(cudaMemPrefetchAsync(ptr, size, Constants::DEVICE_ID));
  }

  for (auto ptr : MemoryManager::managedMemoryAddresses) {
    if (!visited[ptr]) {
      visited[ptr] = true;
      checkCudaErrors(cudaMemPrefetchAsync(ptr, MemoryManager::managedMemoryAddressToSizeMap[ptr], Constants::DEVICE_ID));
    }
  }

  checkCudaErrors(cudaDeviceSynchronize());
}

void executeOptimizedGraph(CustomGraph& optimizedGraph) {
  LOG_TRACE();
  TaskManager::getInstance()->executeOptimizedGraph(optimizedGraph);
}
