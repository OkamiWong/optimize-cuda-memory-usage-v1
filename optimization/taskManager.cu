#include <cuda.h>

#include <map>
#include <memory>
#include <vector>

#include "../utilities/cudaUtilities.hpp"
#include "taskManager.hpp"

TaskManager *TaskManager::instance = nullptr;

TaskManager *TaskManager::getInstance() {
  if (instance == nullptr) {
    instance = new TaskManager();
  }
  return instance;
}

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
}

std::map<GraphNodeId, float> TaskManager::getKernelRunningTimes(cudaGraph_t graph) {
  // Extract nodes and edges

  std::map<GraphNodeId, float> kernelRunningTimes;
  return kernelRunningTimes;
}
