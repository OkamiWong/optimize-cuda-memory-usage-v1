#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "cudaGraphUtilities.hpp"
#include "cudaUtilities.hpp"

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges
) {
  size_t numNodes, numEdges;
  checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
  checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  auto rawNodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
  auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
  auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
  checkCudaErrors(cudaGraphGetNodes(graph, rawNodes.get(), &numNodes));
  checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

  nodes.clear();
  for (int i = 0; i < numNodes; i++) {
    nodes.push_back(rawNodes[i]);
  }

  edges.clear();
  for (int i = 0; i < numEdges; i++) {
    edges[from[i]].push_back(to[i]);
  }
}

cudaGraphNode_t getRootNode(cudaGraph_t graph) {
  size_t numRootNodes;
  checkCudaErrors(cudaGraphGetRootNodes(graph, nullptr, &numRootNodes));
  assert(numRootNodes == 1);

  auto rootNodes = std::make_unique<cudaGraphNode_t[]>(numRootNodes);
  checkCudaErrors(cudaGraphGetRootNodes(graph, rootNodes.get(), &numRootNodes));
  return rootNodes[0];
}
