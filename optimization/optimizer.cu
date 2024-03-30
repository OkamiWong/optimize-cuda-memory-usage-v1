#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

#include "../profiling/annotation.hpp"
#include "../profiling/cudaGraphExecutionTimelineProfiler.hpp"
#include "../profiling/memoryManager.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/disjointSet.hpp"
#include "../utilities/logger.hpp"
#include "optimizer.hpp"
#include "strategies/strategies.hpp"

namespace memopt {

static CUfunction dummyKernelFuncHandle;

void registerDummyKernelFuncHandle(cudaGraph_t graph) {
  // The graph is assumed to have only one root
  // and that root is supposed to be an annotation node
  CUDA_KERNEL_NODE_PARAMS rootNodeParams;
  getKernelNodeParams(getRootNode(graph), rootNodeParams);
  dummyKernelFuncHandle = rootNodeParams.func;
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
  if (nodeToAnnotationMap.find(currentNode) != nodeToAnnotationMap.end()) {
    return;
  }

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

      if (nodeParams.func == dummyKernelFuncHandle) {
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

OptimizationInput::TaskGroup::DataDependency convertTaskAnnotationToTaskGroupDataDependency(
  const TaskAnnotation &taskAnnotation
) {
  OptimizationInput::TaskGroup::DataDependency dep;
  for (int i = 0; i < TaskAnnotation::MAX_NUM_PTR; i++) {
    void *ptr = taskAnnotation.inputs[i];
    if (ptr == nullptr) break;
    dep.inputs.insert(ptr);
  }
  for (int i = 0; i < TaskAnnotation::MAX_NUM_PTR; i++) {
    void *ptr = taskAnnotation.outputs[i];
    if (ptr == nullptr) break;
    dep.outputs.insert(ptr);
  }
  return dep;
}

TaskId getTaskId(cudaGraphNode_t annotationNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(annotationNode, nodeParams);
  assert(nodeParams.func == dummyKernelFuncHandle);

  auto taskAnnotationPtr = reinterpret_cast<TaskAnnotation *>(nodeParams.kernelParams[0]);
  return taskAnnotationPtr->taskId;
}

void mergeDataDependency(OptimizationInput::TaskGroup &taskGroup, cudaGraphNode_t annotationNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(annotationNode, nodeParams);
  assert(nodeParams.func == dummyKernelFuncHandle);

  auto taskAnnotationPtr = reinterpret_cast<TaskAnnotation *>(nodeParams.kernelParams[0]);
  auto taskDataDependency = convertTaskAnnotationToTaskGroupDataDependency(*taskAnnotationPtr);

  taskGroup.dataDependency.inputs.insert(taskDataDependency.inputs.begin(), taskDataDependency.inputs.end());
  taskGroup.dataDependency.outputs.insert(taskDataDependency.outputs.begin(), taskDataDependency.outputs.end());
}

OptimizationInput constructOptimizationInput(
  cudaGraph_t originalGraph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  LOG_TRACE();

  OptimizationInput optimizationInput;

  std::map<cudaGraphNode_t, TaskGroupId> disjointSetRootToTaskGroupIdMap;

  auto getTaskGroupId = [&](cudaGraphNode_t u) {
    auto uTaskId = getTaskId(nodeToAnnotationMap[u]);
    auto uRoot = disjointSet.findRoot(u);

    size_t uTaskGroupId;
    if (disjointSetRootToTaskGroupIdMap.count(uRoot) == 0) {
      optimizationInput.nodes.emplace_back();
      uTaskGroupId = optimizationInput.nodes.size() - 1;
      disjointSetRootToTaskGroupIdMap[uRoot] = uTaskGroupId;
    } else {
      uTaskGroupId = disjointSetRootToTaskGroupIdMap[uRoot];
    }

    optimizationInput.nodes[uTaskGroupId].nodes.insert(uTaskId);

    return uTaskGroupId;
  };

  // Add nodes and edges for both task groups and tasks
  std::set<std::pair<TaskGroupId, TaskGroupId>> existingEdges;
  for (const auto &[u, destinations] : edges) {
    auto uTaskId = getTaskId(nodeToAnnotationMap[u]);
    auto uTaskGroupId = getTaskGroupId(u);

    for (auto v : destinations) {
      auto vTaskId = getTaskId(nodeToAnnotationMap[v]);
      auto vTaskGroupId = getTaskGroupId(v);
      if (uTaskId == vTaskId) {
        continue;
      } else if (uTaskGroupId == vTaskGroupId) {
        optimizationInput.nodes[uTaskGroupId].edges[uTaskId].push_back(vTaskId);
      } else {
        // Edges between task groups need deduping
        if (existingEdges.count(std::make_pair(uTaskGroupId, vTaskGroupId)) == 0) {
          existingEdges.insert(std::make_pair(uTaskGroupId, vTaskGroupId));
          optimizationInput.edges[uTaskGroupId].push_back(vTaskGroupId);
        }
      }
    }
  }

  // Gather information about tasks
  std::map<TaskId, cudaGraphNode_t> taskIdToAnnotationNodeMap;
  std::map<TaskId, std::vector<std::pair<uint64_t, uint64_t>>> taskIdToNodeLifetimes;
  uint64_t globalMinStart = std::numeric_limits<uint64_t>::max(), globalMaxEnd = 0;
  for (cudaGraphNode_t u : nodes) {
    auto uTaskId = getTaskId(nodeToAnnotationMap[u]);

    if (taskIdToAnnotationNodeMap.count(uTaskId) == 0) {
      taskIdToAnnotationNodeMap[uTaskId] = nodeToAnnotationMap[u];
    }

    // Ignore annotation node
    const bool isAnnotationNode = nodeToAnnotationMap[u] == u;
    if (isAnnotationNode) continue;

    // Ignore mem alloc node and mem free node
    cudaGraphNodeType nodeType;
    checkCudaErrors(cudaGraphNodeGetType(u, &nodeType));
    if (nodeType == cudaGraphNodeTypeMemAlloc || nodeType == cudaGraphNodeTypeMemFree) {
      continue;
    }

    taskIdToNodeLifetimes[uTaskId].push_back(timeline[u]);

    globalMinStart = std::min(globalMinStart, timeline[u].first);
    globalMaxEnd = std::max(globalMaxEnd, timeline[u].second);
  }

  // Add task group running time and data dependency
  for (auto &taskGroup : optimizationInput.nodes) {
    std::vector<std::pair<uint64_t, uint64_t>> nodeLifetimes;
    for (auto taskId : taskGroup.nodes) {
      mergeDataDependency(taskGroup, taskIdToAnnotationNodeMap[taskId]);
      nodeLifetimes.insert(nodeLifetimes.end(), taskIdToNodeLifetimes[taskId].begin(), taskIdToNodeLifetimes[taskId].end());
    }

    std::sort(nodeLifetimes.begin(), nodeLifetimes.end());

    uint64_t accumulatedRunningTime = 0;
    uint64_t currentWindowEnd = 0;
    for (auto nodeLifetime : nodeLifetimes) {
      if (currentWindowEnd < nodeLifetime.first) {
        accumulatedRunningTime += nodeLifetime.second - nodeLifetime.first;
        currentWindowEnd = nodeLifetime.second;
      } else {
        if (currentWindowEnd < nodeLifetime.second) {
          accumulatedRunningTime += nodeLifetime.second - currentWindowEnd;
          currentWindowEnd = nodeLifetime.second;
        }
      }
    }

    taskGroup.runningTime = static_cast<float>(accumulatedRunningTime) * 1e-9f;
  }

  optimizationInput.originalTotalRunningTime = static_cast<float>(globalMaxEnd - globalMinStart) * 1e-9f;

  return optimizationInput;
}

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

OptimizationOutput Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  registerDummyKernelFuncHandle(originalGraph);

  auto timeline = getCudaGraphExecutionTimeline(originalGraph);

  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  std::map<cudaGraphNode_t, cudaGraphNode_t> nodeToAnnotationMap;
  mapNodeToAnnotation(originalGraph, edges, nodeToAnnotationMap);

  DisjointSet<cudaGraphNode_t> disjointSet;

  if (ConfigurationManager::getConfig().mergeConcurrentCudaGraphNodes) {
    mergeConcurrentCudaGraphNodes(timeline, disjointSet, std::numeric_limits<int>::max());
  }

  mergeNodesWithSameAnnotation(nodes, nodeToAnnotationMap, disjointSet);

  auto optimizationInput = constructOptimizationInput(originalGraph, nodes, edges, timeline, disjointSet, nodeToAnnotationMap);

  auto optimizedGraph = this->optimize<TwoStepOptimizationStrategy>(optimizationInput);

  if (optimizedGraph.optimal) {
    return optimizedGraph;
  } else {
    LOG_TRACE_WITH_INFO("Could not find any feasible solution");
    exit(-1);
  }
}

}  // namespace memopt
