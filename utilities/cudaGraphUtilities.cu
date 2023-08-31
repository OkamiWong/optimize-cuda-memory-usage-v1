#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "cudaGraphUtilities.hpp"
#include "cudaUtilities.hpp"

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<CUgraphNode> &nodes,
  std::map<CUgraphNode, std::vector<CUgraphNode>> &edges
) {
  size_t numNodes, numEdges;
  checkCudaErrors(cuGraphGetNodes(graph, nullptr, &numNodes));
  checkCudaErrors(cuGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  auto rawNodes = std::make_unique<CUgraphNode[]>(numNodes);
  auto from = std::make_unique<CUgraphNode[]>(numEdges);
  auto to = std::make_unique<CUgraphNode[]>(numEdges);
  checkCudaErrors(cuGraphGetNodes(graph, rawNodes.get(), &numNodes));
  checkCudaErrors(cuGraphGetEdges(graph, from.get(), to.get(), &numEdges));

  nodes.clear();
  for (int i = 0; i < numNodes; i++) {
    nodes.push_back(rawNodes[i]);
  }

  edges.clear();
  for (int i = 0; i < numEdges; i++) {
    edges[from[i]].push_back(to[i]);
  }
}

CUgraphNode getRootNode(cudaGraph_t graph) {
  size_t numRootNodes;
  checkCudaErrors(cuGraphGetRootNodes(graph, NULL, &numRootNodes));
  assert(numRootNodes == 1);

  auto rootNodes = std::make_unique<CUgraphNode[]>(numRootNodes);
  checkCudaErrors(cuGraphGetRootNodes(graph, rootNodes.get(), &numRootNodes));
  return rootNodes[0];
}
