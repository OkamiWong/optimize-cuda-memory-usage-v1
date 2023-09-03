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
  std::map<CUgraphNode, std::vector<CUgraphNode>> &edges,
  std::map<CUgraphNode, KernelDataDependency> &kernelToDataDependencyMap
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

// Only chain shape graph is supported.
// The strategy is always prefetching data for the next kernel.
CustomGraph PrefetchOnlyStrategy::run(
  cudaGraph_t originalGraph,
  Optimizer::CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap
) {
  std::vector<CUgraphNode> nodes;
  std::map<CUgraphNode, std::vector<CUgraphNode>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  auto rootNode = getRootNode(originalGraph);

  auto kernelToDataDependencyMap = mapKernelOntoDataDependency(nodes, edges);

  auto kernelsInExecutionOrder = getKernelsInExecutionOrder(rootNode, edges, kernelToDataDependencyMap);

  // Initialize the custom graph
  CustomGraph customGraph;
  customGraph.originalGraph = originalGraph;

  std::map<CUgraphNode, CustomGraph::NodeId> kernelToKernelStartNodeIdMap;
  std::map<CUgraphNode, CustomGraph::NodeId> kernelToKernelNodeIdMap;
  for (auto &kernel : kernelsInExecutionOrder) {
    const auto kernelStartNodeId = customGraph.addEmptyNode();
    const auto kernelNodeId = customGraph.addKernelNode(kernel);

    customGraph.addEdge(kernelStartNodeId, kernelNodeId);

    kernelToKernelStartNodeIdMap[kernel] = kernelStartNodeId;
    kernelToKernelNodeIdMap[kernel] = kernelNodeId;
  }

  for (int i = 1; i < kernelsInExecutionOrder.size() - 1; i++) {
    auto &previousKernel = kernelsInExecutionOrder[i - 1];
    auto &kernel = kernelsInExecutionOrder[i];
    customGraph.addEdge(
      kernelToKernelNodeIdMap[previousKernel],
      kernelToKernelStartNodeIdMap[kernel]
    );
  }

  // Schedule prefetches
  for (int i = 0; i < kernelsInExecutionOrder.size() - 1; i++) {
    auto &currentKernel = kernelsInExecutionOrder[i];
    auto &nextKernel = kernelsInExecutionOrder[i + 1];

    auto currentKernelStartNodeId = kernelToKernelStartNodeIdMap[currentKernel];
    auto nextKernelStartNodeId = kernelToKernelStartNodeIdMap[nextKernel];

    auto &dataDependency = kernelToDataDependencyMap[nextKernel];

    for (auto &[ptr, size] : dataDependency.inputs) {
      CustomGraph::DataMovement dataMovement;
      dataMovement.direction = CustomGraph::DataMovement::Direction::hostToDevice;
      dataMovement.address = ptr;
      dataMovement.size = size;

      const auto prefetchingNodeId = customGraph.addDataMovementNode(dataMovement);
      customGraph.addEdge(currentKernelStartNodeId, prefetchingNodeId);
      customGraph.addEdge(prefetchingNodeId, nextKernelStartNodeId);
    }

    for (auto &[ptr, size] : dataDependency.outputs) {
      CustomGraph::DataMovement dataMovement;
      dataMovement.direction = CustomGraph::DataMovement::Direction::hostToDevice;
      dataMovement.address = ptr;
      dataMovement.size = size;

      const auto prefetchingNodeId = customGraph.addDataMovementNode(dataMovement);
      customGraph.addEdge(currentKernelStartNodeId, prefetchingNodeId);
      customGraph.addEdge(prefetchingNodeId, nextKernelStartNodeId);
    }
  }

  return customGraph;
}
