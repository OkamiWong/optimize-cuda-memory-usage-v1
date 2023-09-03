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
  const std::map<CUgraphNode, std::vector<CUgraphNode>> &edges,
  const std::map<CUgraphNode, KernelDataDependency> &kernelToDataDependencyMap
) {
  std::vector<CUgraphNode> kernelsInExecutionOrder;

  auto currentNode = rootNode;
  for (;;) {
    assert(edges[currentNode].size() <= 1);

    // Get rid of dummy kernels, such as dummyKernelForAnnotation.
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

  // Initialize data movement plan;
  Optimizer::DataMovementPlan dataMovementPlan;
  dataMovementPlan.originalGraph = originalGraph;

  // Schedule prefetches: always prefetch for the next kernel
  std::vector<std::vector<Optimizer::DataMovementPlan::DataMovementStep>>
    dataMovementStepsPerKernel(kernelsInExecutionOrder.size());
  for (int i = 0; i < kernelsInExecutionOrder.size() - 1; i++) {
    auto &currentKernel = kernelsInExecutionOrder[i];
    auto &nextKernel = kernelsInExecutionOrder[i + 1];
    auto &dataDependency = kernelToDataDependencyMap[nextKernel];

    for (auto &[ptr, size] : dataDependency.inputs) {
      Optimizer::DataMovementPlan::DataMovementStep step;
      step.dataMovement.direction = CustomGraph::DataMovement::Direction::hostToDevice;
      step.dataMovement.address = ptr;
      step.dataMovement.size = size;
      step.dataMovementPosition = currentKernel;
      step.dataMovementRelativePosition = Optimizer::DataMovementPlan::DataMovementRelativePosition::beforeKernel;
      dataMovementStepsPerKernel[i].push_back(step);
    }

    for (auto &[ptr, size] : dataDependency.outputs) {
      Optimizer::DataMovementPlan::DataMovementStep step;
      step.dataMovement.direction = CustomGraph::DataMovement::Direction::hostToDevice;
      step.dataMovement.address = ptr;
      step.dataMovement.size = size;
      step.dataMovementPosition = currentKernel;
      step.dataMovementRelativePosition = Optimizer::DataMovementPlan::DataMovementRelativePosition::beforeKernel;
      dataMovementStepsPerKernel[i].push_back(step);
    }
  }

  return dataMovementPlan;
}
