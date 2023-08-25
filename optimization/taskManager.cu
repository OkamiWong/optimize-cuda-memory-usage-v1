#include <cuda.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
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
  for (int i = 0; i < numEdges; i++) {
    edges[from[i]].push_back(to[i]);
  }
}

void executeNode(CUgraphNode node) {
  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(node, &nodeType));
  if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
    // TODO
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
    // TODO
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEM_FREE) {
    // TODO
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMSET) {
    // TODO
  } else {
    LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
    exit(-1);
  }
}

std::map<GraphNodeId, float> TaskManager::getKernelRunningTimes(cudaGraph_t graph) {
  std::vector<CUgraphNode> nodes;
  std::map<CUgraphNode, std::vector<CUgraphNode>> edges;
  extractGraphNodesAndEdges(graph, nodes, edges);

  std::map<CUgraphNode, int> inDegrees;
  for (auto &[u, outEdges] : edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  typedef std::pair<int, CUgraphNode> RemainingNode;
  std::priority_queue<RemainingNode, std::vector<RemainingNode>, std::greater<RemainingNode>> remainingNodes;
  for (auto &node : nodes) {
    remainingNodes.push(std::make_pair(inDegrees[node], node));
  }

  // Kahn Algorithm
  std::map<GraphNodeId, float> kernelRunningTimes;
  CudaEventClock clock;

  while (!remainingNodes.empty()) {
    auto [inDegree, u] = remainingNodes.top();
    assert(inDegree == 0);
    remainingNodes.pop();

    clock.start();
    executeNode(u);
    clock.end();
    checkCudaErrors(cudaDeviceSynchronize());
    kernelRunningTimes[reinterpret_cast<GraphNodeId>(u)] = clock.getTimeInSeconds();
  }

  return kernelRunningTimes;
}
