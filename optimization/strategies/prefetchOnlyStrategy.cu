#include <cassert>
#include <map>
#include <vector>

#include "../../utilities/cudaGraphUtilities.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

// Currently, only support chain shape graph.
Optimizer::DataMovementPlan PrefetchOnlyStrategy::calculateDataMovementPlan(
  cudaGraph_t originalGraph,
  Optimizer::CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap
) {
  std::vector<CUgraphNode> nodes;
  std::map<CUgraphNode, std::vector<CUgraphNode>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  auto rootNode = getRootNode(originalGraph);

  auto kernelToDataDependencyMap = mapKernelOntoDataDependency(nodes, edges);

  // Sort kernels by execution order
  std::vector<CUgraphNode> kernels;
  auto currentNode = rootNode;
  for (;;) {
    assert(edges[currentNode].size() <= 1);

    if (kernelToDataDependencyMap.find(currentNode) != kernelToDataDependencyMap.end()) {
      kernels.push_back(currentNode);
    }

    if (edges[currentNode].size() == 1) {
      currentNode = edges[currentNode][0];
    } else {
      break;
    }
  }

  // Schedule prefetches
  // TODO
}
