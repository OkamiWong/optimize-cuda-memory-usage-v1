#include <cuda.h>

#include <cassert>
#include <limits>
#include <utility>

#include "../profiling/annotation.hpp"
#include "../profiling/memoryManager.hpp"
#include "../utilities/cudaGraphExecutionTimelineProfiler.hpp"
#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/disjointSet.hpp"
#include "../utilities/logger.hpp"
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
  LOG_TRACE();

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
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  const int logicalNodeSizeLimit
) {
  LOG_TRACE();

  std::map<CudaGraphNodeLifetime, cudaGraphNode_t> lifetimeToCudaGraphNodeMap;
  for (auto &[node, lifetime] : timeline) {
    lifetimeToCudaGraphNodeMap[lifetime] = node;
  }

  uint64_t currentWindowEnd = 0;
  cudaGraphNode_t currentWindowRepresentativeNode = nullptr;
  for (auto &[lifetime, node] : lifetimeToCudaGraphNodeMap) {
    // Ignore mem alloc node and mem free node
    cudaGraphNodeType nodeType;
    checkCudaErrors(cudaGraphNodeGetType(node, &nodeType));
    if (nodeType == cudaGraphNodeTypeMemAlloc || nodeType == cudaGraphNodeTypeMemFree) {
      continue;
    }

    assert(lifetime.first != 0 && lifetime.second != 0);

    if (currentWindowRepresentativeNode != nullptr && lifetime.first <= currentWindowEnd && disjointSet.getSetSize(currentWindowRepresentativeNode) < logicalNodeSizeLimit) {
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
  cudaGraphNode_t currentAnnotationNode,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  bool isAnnotationNode = false;

  if (!currentAnnotationNode) {
    isAnnotationNode = true;
  } else {
    cudaGraphNodeType nodeType;
    checkCudaErrors(cudaGraphNodeGetType(currentNode, &nodeType));
    if (nodeType == cudaGraphNodeTypeKernel) {
      // Why switch to driver API:
      // https://forums.developer.nvidia.com/t/cuda-runtime-api-error-for-cuda-graph-and-opencv/215408/13
      CUDA_KERNEL_NODE_PARAMS nodeParams;
      checkCudaErrors(cuGraphKernelNodeGetParams(currentNode, &nodeParams));

      if (nodeParams.func == TaskManager::getInstance()->getDummyKernelHandle()) {
        isAnnotationNode = true;
      }
    }
  }

  if (isAnnotationNode) {
    currentAnnotationNode = currentNode;
  }

  nodeToAnnotationMap[currentNode] = currentAnnotationNode;

  for (auto nextNode : edges[currentNode]) {
    dfs(nextNode, currentAnnotationNode, edges, nodeToAnnotationMap);
  }
}

void mapNodeToAnnotation(
  cudaGraph_t originalGraph,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  LOG_TRACE();

  auto rootNode = getRootNode(originalGraph);
  dfs(rootNode, nullptr, edges, nodeToAnnotationMap);
}

void mergeNodesWithSameAnnotation(
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  DisjointSet<cudaGraphNode_t> &disjointSet
) {
  for (auto u : nodes) {
    disjointSet.unionUnderlyingSets(u, nodeToAnnotationMap[u]);
  }
}

OptimizationInput::LogicalNode::DataDependency
convertKernelIOToKernelDataDependency(const KernelIO &kernelIO) {
  OptimizationInput::LogicalNode::DataDependency dep;
  for (int i = 0; i < KernelIO::MAX_NUM_PTR; i++) {
    void *ptr = kernelIO.inputs[i];
    if (ptr == nullptr) break;
    dep.inputs.insert(ptr);
  }
  for (int i = 0; i < KernelIO::MAX_NUM_PTR; i++) {
    void *ptr = kernelIO.outputs[i];
    if (ptr == nullptr) break;
    dep.outputs.insert(ptr);
  }
  return dep;
}

void mergeDataDependency(OptimizationInput::LogicalNode &logicalNode, cudaGraphNode_t annotationNode) {
  cudaGraphNodeType nodeType;
  checkCudaErrors(cudaGraphNodeGetType(annotationNode, &nodeType));
  assert(nodeType == cudaGraphNodeTypeKernel);

  // Why switch to driver API:
  // https://forums.developer.nvidia.com/t/cuda-runtime-api-error-for-cuda-graph-and-opencv/215408/13
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  checkCudaErrors(cuGraphKernelNodeGetParams(annotationNode, &nodeParams));

  assert(nodeParams.func == TaskManager::getInstance()->getDummyKernelHandle());

  auto kernelIOPtr = reinterpret_cast<KernelIO *>(nodeParams.kernelParams[0]);
  auto dataDependencyByAnnotation = convertKernelIOToKernelDataDependency(*kernelIOPtr);

  logicalNode.dataDependency.inputs.insert(dataDependencyByAnnotation.inputs.begin(), dataDependencyByAnnotation.inputs.end());
  logicalNode.dataDependency.outputs.insert(dataDependencyByAnnotation.outputs.begin(), dataDependencyByAnnotation.outputs.end());
}

OptimizationInput constructOptimizationInput(
  cudaGraph_t originalGraph,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  LOG_TRACE();

  OptimizationInput optimizationInput;

  std::map<cudaGraphNode_t, OptimizationInput::NodeId> disjointSetRootToLogicalNodeIndexMap;

  auto getLogicalNodeId = [&](cudaGraphNode_t u) {
    auto uRoot = disjointSet.findRoot(u);

    size_t uLogicalNodeId;
    if (disjointSetRootToLogicalNodeIndexMap.find(uRoot) == disjointSetRootToLogicalNodeIndexMap.end()) {
      optimizationInput.nodes.emplace_back();
      uLogicalNodeId = optimizationInput.nodes.size() - 1;
      disjointSetRootToLogicalNodeIndexMap[uRoot] = uLogicalNodeId;
    } else {
      uLogicalNodeId = disjointSetRootToLogicalNodeIndexMap[uRoot];
    }

    optimizationInput.nodes[uLogicalNodeId].nodes.insert(u);

    return uLogicalNodeId;
  };

  // Add nodes and edges, both logical nodes and actual nodes
  std::set<std::pair<OptimizationInput::NodeId, OptimizationInput::NodeId>> existingEdges;
  for (const auto &[u, destinations] : edges) {
    auto uLogicalNodeId = getLogicalNodeId(u);

    for (auto v : destinations) {
      auto vLogicalNodeId = getLogicalNodeId(v);
      if (uLogicalNodeId == vLogicalNodeId) {
        optimizationInput.nodes[uLogicalNodeId].edges[u].push_back(v);
      } else {
        // Logical node edges needs deduping
        if (existingEdges.count(std::make_pair(uLogicalNodeId, vLogicalNodeId)) == 0) {
          existingEdges.insert(std::make_pair(uLogicalNodeId, vLogicalNodeId));
          optimizationInput.edges[uLogicalNodeId].push_back(vLogicalNodeId);
        }
      }
    }
  }

  // Add duration and data dependency
  uint64_t globalMinStart = std::numeric_limits<uint64_t>::max(), globalMaxEnd = 0;
  for (auto &logicalNode : optimizationInput.nodes) {
    uint64_t minStart = std::numeric_limits<uint64_t>::max(), maxEnd = 0;

    for (auto node : logicalNode.nodes) {
      // Ignore annotation node
      const auto isAnnotationNode = nodeToAnnotationMap[node] == node;
      if (isAnnotationNode) continue;

      // Ignore mem alloc node and mem free node
      cudaGraphNodeType nodeType;
      checkCudaErrors(cudaGraphNodeGetType(node, &nodeType));
      if (nodeType == cudaGraphNodeTypeMemAlloc || nodeType == cudaGraphNodeTypeMemFree) {
        continue;
      }

      minStart = std::min(minStart, timeline[node].first);
      maxEnd = std::max(maxEnd, timeline[node].second);

      mergeDataDependency(logicalNode, nodeToAnnotationMap[node]);
    }

    globalMinStart = std::min(globalMinStart, minStart);
    globalMaxEnd = std::max(globalMaxEnd, maxEnd);

    logicalNode.duration = static_cast<float>(maxEnd - minStart) * 1e-9f;
  }

  optimizationInput.originalTotalRunningTime = static_cast<float>(globalMaxEnd - globalMinStart) * 1e-9f;

  return optimizationInput;
}

CustomGraph Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  // Profile
  auto taskManager = TaskManager::getInstance();
  taskManager->registerDummyKernelHandle(originalGraph);

  auto timeline = getCudaGraphExecutionTimeline(originalGraph);

  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  bool feasibleSolutionFound = false;
  CustomGraph lastFeasibleGraph;
  int logicalNodeSizeLimit = std::numeric_limits<int>::max();
  for (;;) {
    LOG_TRACE_WITH_INFO("logicalNodeSizeLimit = %d", logicalNodeSizeLimit);

    DisjointSet<cudaGraphNode_t> disjointSet;
    mergeConcurrentCudaGraphNodes(timeline, disjointSet, logicalNodeSizeLimit);

    int largestLogicalNodeSize = 0;
    for (auto node : nodes) {
      largestLogicalNodeSize = std::max(largestLogicalNodeSize, static_cast<int>(disjointSet.getSetSize(node)));
    }

    std::map<cudaGraphNode_t, cudaGraphNode_t> nodeToAnnotationMap;
    mapNodeToAnnotation(originalGraph, edges, nodeToAnnotationMap);

    mergeNodesWithSameAnnotation(nodes, nodeToAnnotationMap, disjointSet);

    auto optimizationInput = constructOptimizationInput(originalGraph, edges, timeline, disjointSet, nodeToAnnotationMap);

    // Optimize
    auto optimizedGraph = this->optimize<TwoStepOptimizationStrategy>(optimizationInput);

    if (optimizedGraph.optimal) {
      feasibleSolutionFound = true;
      lastFeasibleGraph = optimizedGraph;

      logicalNodeSizeLimit = (largestLogicalNodeSize + 1) / 2;

      if (largestLogicalNodeSize == 1) {
        return optimizedGraph;
      }
    } else {
      if (feasibleSolutionFound) {
        return lastFeasibleGraph;
      }
      exit(-1);
    }
  }
}
