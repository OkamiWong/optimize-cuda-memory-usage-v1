#include <cassert>

#include "../utilities/cudaGraphExecutionTimelineProfiler.hpp"
#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/disjointSet.hpp"
#include "optimizer.hpp"
#include "strategies/strategies.hpp"
#include "taskManager.hpp"

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

CudaGraphExecutionTimeline getCudaGraphExecutionTimeline(cudaGraph_t graph) {
  auto profiler = CudaGraphExecutionTimelineProfiler::getInstance();
  profiler->initialize(graph);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaDeviceSynchronize());

  profiler->finalize();

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaStreamDestroy(stream));

  return profiler->getTimeline();
}

void mergeConcurrentCudaGraphNodes(
  const CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet
) {
  std::map<CudaGraphNodeLifetime, cudaGraphNode_t> lifetimeToCudaGraphNodeMap;
  for (auto &[node, lifetime] : timeline) {
    lifetimeToCudaGraphNodeMap[lifetime] = node;
  }

  uint64_t currentWindowEnd = 0;
  cudaGraphNode_t currentWindowRepresentativeNode = nullptr;
  for (auto &[lifetime, node] : lifetimeToCudaGraphNodeMap) {
    assert(lifetime.first != 0 && lifetime.second != 0);

    if (currentWindowRepresentativeNode != nullptr && lifetime.first <= currentWindowEnd) {
      disjointSet.unionUnderlyingSets(currentWindowRepresentativeNode, node);
      currentWindowEnd = std::max(currentWindowEnd, lifetime.second);
    } else {
      currentWindowRepresentativeNode = node;
      currentWindowEnd = lifetime.second;
    }
  }
}

void dfs(
  cudaGraphNode_t currentNode,
  cudaGraphNode_t parent,
  const std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  DisjointSet<cudaGraphNode_t> &disjointSet
) {
  bool isAnnotationNode = false;

  if (!parent) {
    isAnnotationNode = true;
  } else {
    cudaGraphNodeType nodeType;
    checkCudaErrors(cudaGraphNodeGetType(currentNode, &nodeType));
    if (nodeType == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams nodeParams;
      checkCudaErrors(cudaGraphKernelNodeGetParams(currentNode, &nodeParams));
      if (nodeParams.func == TaskManager::getInstance()->getDummyKernelHandle()) {
        isAnnotationNode = true;
      }
    }
  }

  if (!isAnnotationNode) {
    disjointSet.unionUnderlyingSets(currentNode, parent);
  }

  for (auto nextNode : edges[currentNode]) {
    dfs(nextNode, currentNode, edges, disjointSet);
  }
}

void mergeCudaGraphNodesWithSameAnnotation(
  cudaGraphNode_t rootNode,
  const std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  DisjointSet<cudaGraphNode_t> &disjointSet
) {
  auto rootNode = getRootNode(originalGraph);
  dfs(rootNode, nullptr, edges, disjointSet);
}

OptimizationInput constructOptimizationInput(cudaGraph_t originalGraph, const CudaGraphExecutionTimeline &timeline, const DisjointSet<cudaGraphNode_t> &disjointSet) {
}

CustomGraph Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  // Profile
  auto taskManager = TaskManager::getInstance();
  taskManager->registerDummyKernelHandle(originalGraph);

  auto timeline = getCudaGraphExecutionTimeline(originalGraph);

  DisjointSet<cudaGraphNode_t> disjointSet;
  mergeConcurrentCudaGraphNodes(timeline, disjointSet);

  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  mergeCudaGraphNodesWithSameAnnotation(getRootNode(originalGraph), edges, disjointSet);
  auto optimizationInput = constructOptimizationInput(originalGraph, timeline, disjointSet);

  // Optimize
  auto customGraph = this->optimize<TwoStepOptimizationStrategy>(optimizationInput);
  return customGraph;
}
