#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <vector>

#include "../../profiling/memoryManager.hpp"
#include "../../utilities/cudaGraphUtilities.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

std::vector<CUgraphNode> getKernelsInExecutionOrder(
  CUgraphNode rootNode,
  std::map < CUgraphNode, std::vector<CUgraphNode>,
  std::map<CUgraphNode, KernelDataDependency> kernelToDataDependencyMap
) {
  std::vector<CUgraphNode> kernelsInExecutionOrder;

  auto currentNode = rootNode;
  for (;;) {
    assert(edges[currentNode].size() <= 1);

    if (kernelToDataDependencyMap.find(currentNode) != kernelToDataDependencyMap.end()) {
      kernelsInExecutionOrder.push_back(currentNode);
    }

    if (edges[currentNode].size() == 1) {
      currentNode = edges[currentNode][0];
    } else {
      break;
    }
  }

  return kernelsInExecutionOrder;
}

std::map<void *, bool> getMemorySpaceOnDeviceOrNotMap(
  std::vector<CUgraphNode> kernels,
  std::map<CUgraphNode, KernelDataDependency> kernelToDataDependencyMap
) {
  std::map<void *, bool> memorySpaceOnDeviceOrNotMap;
  for (auto &[ptr, size] : MemoryManager::managedMemorySpaces) {
    memorySpaceOnDeviceOrNotMap[ptr] = false;
  }
  for (auto &k : kernels) {
    auto &dataDependency = kernelToDataDependencyMap[k];
    bool stop = false;
    for (auto &[ptr, size] : dataDependency.inputs) {
      if (MemoryManager::managedMemorySpacesInitiallyOnDevice.count(ptr) == 1) {
        memorySpaceOnDeviceOrNotMap[ptr] = true;
      } else {
        stop = true;
        break;
      }
    }
    for (auto &[ptr, size] : dataDependency.outputs) {
      if (MemoryManager::managedMemorySpacesInitiallyOnDevice.count(ptr) == 1) {
        memorySpaceOnDeviceOrNotMap[ptr] = true;
      } else {
        stop = true;
        break;
      }
    }
    if (stop) {
      break;
    }
  }
  return memorySpaceOnDeviceOrNotMap;
}

// Currently, only support chain shape graph.
Optimizer::DataMovementPlan
PrefetchOnlyStrategy::calculateDataMovementPlan(
  cudaGraph_t originalGraph,
  Optimizer::CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap
) {
  std::vector<CUgraphNode> nodes;
  std::map<CUgraphNode, std::vector<CUgraphNode>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  auto rootNode = getRootNode(originalGraph);

  auto kernelToDataDependencyMap = mapKernelOntoDataDependency(nodes, edges);

  auto kernelsInExecutionOrder = getKernelsInExecutionOrder(rootNode, edges, kernelToDataDependencyMap);

  auto memorySpaceOnDeviceOrNotMap = getMemorySpaceOnDeviceOrNotMap(kernelsInExecutionOrder, kernelToDataDependencyMap);

  // Initialize data movement plan;
  Optimizer::DataMovementPlan dataMovementPlan;
  dataMovementPlan.originalGraph = originalGraph;

  // Schedule prefetches
  size_t currentMemoryUsed = 0;
  float currentPrefetchingFrontline = 0;
  float currentComputingFrontline = 0;
  int currentKernelIndex = 0;
  int current
  while (currentKernelIndex < kernelsInExecutionOrder.size()) {
    auto &k = kernelsInExecutionOrder[currentKernelIndex];
    auto &dataDependency = kernelToDataDependencyMap[k];

    std::vector<MemoryManager::ArrayInfo> unsatisfiedDataDependencies;
    for (auto &[ptr, size] : dataDependency.inputs) {
      if (!memorySpaceOnDeviceOrNotMap[ptr]) {
        unsatisfiedDataDependencies.push_back(std::make_tuple(ptr, size));
      }
    }
    for (auto &[ptr, size] : dataDependency.outputs) {
      if (!memorySpaceOnDeviceOrNotMap[ptr]) {
        unsatisfiedDataDependencies.push_back(std::make_tuple(ptr, size));
      }
    }

    if (unsatisfiedDataDependencies.empty()) {
      currentComputingFrontline += cuGraphNodeToKernelDurationMap[k];
      currentKernelIndex++;
    } else {
      for (auto &[ptr, size] : unsatisfiedDataDependencies) {
        currentPrefetchingFrontline += static_cast<float>(size) / CudaConstants::PREFETCHING_BANDWIDTH;
        Optimizer::DataMovementPlan::DataMovementStep step;
        // TODO: Add the step to the plan
        // TODO: Consider memory capacity limit
      }
      currentComputingFrontline = std::max(currentPrefetchingFrontline, currentComputingFrontline);
      currentComputingFrontline += CuGraphNodeToKernelDurationMap[k];
    }
  }

  return dataMovementPlan;
}
