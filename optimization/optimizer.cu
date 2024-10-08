#include <cuda.h>

#include <algorithm>
#include <cassert>
#include <exception>
#include <limits>
#include <utility>

#include "../profiling/annotation.hpp"
#include "../profiling/cudaGraphExecutionTimelineProfiler.hpp"
#include "../profiling/memoryManager.hpp"
#include "../profiling/memred.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/disjointSet.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"
#include "optimizer.hpp"
#include "strategies/strategies.hpp"

namespace memopt {

static CUfunction dummyKernelForAnnotationHandle;
static CUfunction dummyKernelForStageSeparatorHandle;

cudaGraph_t dummyKernelForAnnotationGraph;
cudaGraph_t dummyKernelForStageSeparatorGraph;

void registerDummyKernelHandles() {
  cudaStream_t s;
  cudaStreamCreate(&s);

  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  TaskAnnotation a{};
  dummyKernelForAnnotation<<<1, 1, 0, s>>>(a);
  cudaStreamEndCapture(s, &(dummyKernelForAnnotationGraph));
  CUDA_KERNEL_NODE_PARAMS rootNodeParams;
  getKernelNodeParams(getRootNode(dummyKernelForAnnotationGraph), rootNodeParams);
  dummyKernelForAnnotationHandle = rootNodeParams.func;

  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  dummyKernelForStageSeparator<<<1, 1, 0, s>>>();
  cudaStreamEndCapture(s, &(dummyKernelForStageSeparatorGraph));
  getKernelNodeParams(getRootNode(dummyKernelForStageSeparatorGraph), rootNodeParams);
  dummyKernelForStageSeparatorHandle = rootNodeParams.func;

  checkCudaErrors(cudaStreamDestroy(s));
}

void cleanUpDummyKernelFuncHandleRegistrations() {
  checkCudaErrors(cudaGraphDestroy(dummyKernelForAnnotationGraph));
  checkCudaErrors(cudaGraphDestroy(dummyKernelForStageSeparatorGraph));
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

void mapNodeToAnnotationDfs(
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
  } else if (compareKernelNodeFunctionHandle(currentNode, dummyKernelForAnnotationHandle)) {
    isAnnotationNode = true;
  }

  if (isAnnotationNode) {
    currentAnnotationNode = currentNode;
  }

  nodeToAnnotationMap[currentNode] = currentAnnotationNode;

  for (auto nextNode : edges[currentNode]) {
    mapNodeToAnnotationDfs(nextNode, currentAnnotationNode, edges, nodeToAnnotationMap);
  }
}

void mapNodeToAnnotation(
  cudaGraph_t originalGraph,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap
) {
  LOG_TRACE();

  auto rootNode = getRootNode(originalGraph);
  mapNodeToAnnotationDfs(rootNode, nullptr, edges, nodeToAnnotationMap);
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

TaskId getTaskId(cudaGraphNode_t annotationNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(annotationNode, nodeParams);
  assert(nodeParams.func == dummyKernelForAnnotationHandle);

  auto taskAnnotationPtr = reinterpret_cast<TaskAnnotation *>(nodeParams.kernelParams[0]);
  return taskAnnotationPtr->taskId;
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

OptimizationInput::TaskGroup::DataDependency getDataDependencyFromAnnotationNode(cudaGraphNode_t annotationNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  getKernelNodeParams(annotationNode, nodeParams);
  assert(nodeParams.func == dummyKernelForAnnotationHandle);

  auto taskAnnotationPtr = reinterpret_cast<TaskAnnotation *>(nodeParams.kernelParams[0]);
  return convertTaskAnnotationToTaskGroupDataDependency(*taskAnnotationPtr);
}

void mergeDataDependency(OptimizationInput::TaskGroup::DataDependency &dataDependency, const OptimizationInput::TaskGroup::DataDependency &additionalDataDependency) {
  dataDependency.inputs.insert(additionalDataDependency.inputs.begin(), additionalDataDependency.inputs.end());
  dataDependency.outputs.insert(additionalDataDependency.outputs.begin(), additionalDataDependency.outputs.end());
}

void getTaskDataDependencies(
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> &taskIdToDataDependencyMap
) {
  MemRedAnalysisParser analysisParser;

  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> annotationToNodesMap;
  for (const auto [node, annotationNode] : nodeToAnnotationMap) {
    annotationToNodesMap[annotationNode].push_back(node);
  }

  for (auto [annotationNode, nodes] : annotationToNodesMap) {
    const TaskId taskId = getTaskId(annotationNode);
    auto dataDependency = getDataDependencyFromAnnotationNode(annotationNode);

    if (dataDependency.inputs.empty() && dataDependency.outputs.empty()) {
      // This task's data dependency should be inferred from compiler pass
      // and kernel parameters.

      for (auto node : nodes) {
        auto nodeType = getNodeType(node);

        // Currently, only kernel's data dependency can be analyzed automatically.
        if (nodeType == cudaGraphNodeTypeKernel) {
          // Skip dummy kernels
          if (!compareKernelNodeFunctionHandle(node, dummyKernelForAnnotationHandle)
              && !compareKernelNodeFunctionHandle(node, dummyKernelForStageSeparatorHandle)) {
            mergeDataDependency(dataDependency, analysisParser.getKernelDataDependency(node));
          }
        }
      }
    }

    taskIdToDataDependencyMap[taskId] = dataDependency;
  }
}

void mapNodeToStageDfs(
  cudaGraphNode_t currentNode,
  int currentStageIndex,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, int> &nodeToStageIndexMap
) {
  if (nodeToStageIndexMap.find(currentNode) != nodeToStageIndexMap.end()) {
    return;
  }

  bool isStageSeparatorNode = compareKernelNodeFunctionHandle(currentNode, dummyKernelForStageSeparatorHandle);

  // The separator node belongs to the previous stage
  nodeToStageIndexMap[currentNode] = currentStageIndex;

  if (isStageSeparatorNode) {
    currentStageIndex++;
  }

  for (auto nextNode : edges[currentNode]) {
    mapNodeToStageDfs(nextNode, currentStageIndex, edges, nodeToStageIndexMap);
  }
}

void mapNodeToStage(
  cudaGraph_t originalGraph,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  std::map<cudaGraphNode_t, int> &nodeToStageIndexMap
) {
  LOG_TRACE();

  auto rootNode = getRootNode(originalGraph);
  mapNodeToStageDfs(rootNode, 0, edges, nodeToStageIndexMap);
}

OptimizationInput constructOptimizationInput(
  cudaGraph_t originalGraph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges,
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> &taskIdToDataDependencyMap
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
      mergeDataDependency(taskGroup.dataDependency, taskIdToDataDependencyMap[taskId]);
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

  optimizationInput.forceAllArraysToResideOnHostInitiallyAndFinally = false;

  optimizationInput.stageIndex = 0;

  // Print statistics
  double averageTaskGroupRunningTime = 0.0;
  double averageTaskGroupDataDependencySizeInGiB = 0.0;
  double averageTaskGroupProcessingSpeed = 0.0;
  for (const auto &tg : optimizationInput.nodes) {
    averageTaskGroupRunningTime += tg.runningTime;

    size_t s = 0;
    for (auto p : tg.dataDependency.inputs) {
      s += MemoryManager::managedMemoryAddressToSizeMap[p];
    }
    for (auto p : tg.dataDependency.outputs) {
      s += MemoryManager::managedMemoryAddressToSizeMap[p];
    }

    averageTaskGroupDataDependencySizeInGiB += (double)s / 1024.0 / 1024.0 / 1024.0;

    averageTaskGroupProcessingSpeed += (double)s / 1024.0 / 1024.0 / 1024.0 / tg.runningTime;
  }
  averageTaskGroupRunningTime /= optimizationInput.nodes.size();
  averageTaskGroupDataDependencySizeInGiB /= optimizationInput.nodes.size();
  averageTaskGroupProcessingSpeed /= optimizationInput.nodes.size();

  LOG_TRACE_WITH_INFO("Number of task groups: %d", (int)optimizationInput.nodes.size());
  LOG_TRACE_WITH_INFO("Average task group running time (s): %.12lf", averageTaskGroupRunningTime);
  LOG_TRACE_WITH_INFO("Average task group data dependency size (GiB): %.12lf", averageTaskGroupDataDependencySizeInGiB);
  LOG_TRACE_WITH_INFO("Average task group data processing speed (GiB / s): %.12lf", averageTaskGroupProcessingSpeed);

  return optimizationInput;
}

std::vector<OptimizationInput> constructOptimizationInputsForStages(
  cudaGraph_t originalGraph,
  std::vector<cudaGraphNode_t> &allNodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &allEdges,
  CudaGraphExecutionTimeline &timeline,
  DisjointSet<cudaGraphNode_t> &disjointSet,
  std::map<cudaGraphNode_t, cudaGraphNode_t> &nodeToAnnotationMap,
  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> &taskIdToDataDependencyMap,
  std::map<cudaGraphNode_t, int> &nodeToStageIndexMap
) {
  std::vector<std::vector<cudaGraphNode_t>> nodesPerStage;
  for (auto node : allNodes) {
    int stageIndex = nodeToStageIndexMap[node];
    while (nodesPerStage.size() <= stageIndex) nodesPerStage.push_back({});
    nodesPerStage[stageIndex].push_back(node);
  }

  const int numberOfStages = nodesPerStage.size();

  std::vector<std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>>> edgesPerStage(numberOfStages);
  for (int i = 0; i < numberOfStages; i++) {
    for (auto u : nodesPerStage[i]) {
      for (auto v : allEdges[u]) {
        if (nodeToStageIndexMap[v] == i) {
          edgesPerStage[i][u].push_back(v);
        }
      }
    }
  }

  std::vector<OptimizationInput> optimizationInputs;
  for (int i = 0; i < numberOfStages; i++) {
    optimizationInputs.push_back(
      constructOptimizationInput(
        originalGraph, nodesPerStage[i], edgesPerStage[i], timeline, disjointSet, nodeToAnnotationMap, taskIdToDataDependencyMap
      )
    );
    optimizationInputs.rbegin()->forceAllArraysToResideOnHostInitiallyAndFinally = true;
    optimizationInputs.rbegin()->stageIndex = i;
  }
  return optimizationInputs;
}

OptimizationOutput mergeOptimizationOutputs(std::vector<OptimizationOutput> &optimizationOutputs) {
  auto getRootNodeIndex = [](OptimizationOutput &output) {
    std::map<int, bool> hasIncomingEdge;
    for (auto u : output.nodes) {
      for (auto v : output.edges[u]) {
        hasIncomingEdge[v] = true;
      }
    }
    for (auto u : output.nodes) {
      if (!hasIncomingEdge[u]) return u;
    }
    throw std::runtime_error("Cannot find root node");
  };

  auto getLastNodeIndex = [](OptimizationOutput &output) {
    auto u = output.nodes[0];
    while (output.edges[u].size() != 0) u = output.edges[u][0];
    return u;
  };

  OptimizationOutput mergedOptimizationOutput = optimizationOutputs[0];
  int globalLastNodeIndex = getLastNodeIndex(mergedOptimizationOutput);
  int globalNodeIndexOffset = mergedOptimizationOutput.nodes.size();
  for (int i = 1; i < optimizationOutputs.size(); i++) {
    auto &output = optimizationOutputs[i];
    int rootNodeIndex = getRootNodeIndex(output);
    int lastNodeIndex = getLastNodeIndex(output);
    for (auto u : output.nodes) {
      int newU = u + globalNodeIndexOffset;
      mergedOptimizationOutput.nodes.push_back(newU);
      for (auto v : output.edges[u]) {
        mergedOptimizationOutput.edges[newU].push_back(v + globalNodeIndexOffset);
      }

      auto nodeType = output.nodeIdToNodeTypeMap[u];
      mergedOptimizationOutput.nodeIdToNodeTypeMap[newU] = nodeType;
      if (nodeType == OptimizationOutput::NodeType::task) {
        mergedOptimizationOutput.nodeIdToTaskIdMap[newU] = output.nodeIdToTaskIdMap[u];
      } else if (nodeType == OptimizationOutput::NodeType::dataMovement) {
        mergedOptimizationOutput.nodeIdToDataMovementMap[newU] = output.nodeIdToDataMovementMap[u];
      }
    }

    mergedOptimizationOutput.addEdge(globalLastNodeIndex, rootNodeIndex + globalNodeIndexOffset);
    globalLastNodeIndex = lastNodeIndex + globalNodeIndexOffset;
    globalNodeIndexOffset += output.nodes.size();
  }

  mergedOptimizationOutput.arraysInitiallyAllocatedOnDevice.clear();

  return mergedOptimizationOutput;
}

struct SerializableOptimizationOutputNode {
  int nodeId;
  std::vector<int> edges;
  OptimizationOutput::NodeType nodeType;
  int taskId;
  OptimizationOutput::DataMovement::Direction direction;
  int arrayId;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    SerializableOptimizationOutputNode,
    nodeId,
    nodeType,
    edges,
    taskId,
    direction,
    arrayId
  );
};

void writeOptimizationOutputToFile(OptimizationOutput &output, const std::string &path) {
  LOG_TRACE_WITH_INFO("Printing optimization plan to %s", path.c_str());

  std::vector<SerializableOptimizationOutputNode> serializableNodes;
  for (auto i : output.nodes) {
    SerializableOptimizationOutputNode node;
    node.nodeId = i;
    node.edges = output.edges[i];
    node.nodeType = output.nodeIdToNodeTypeMap[i];
    if (node.nodeType == OptimizationOutput::NodeType::task) {
      node.taskId = output.nodeIdToTaskIdMap[i];
    } else if (node.nodeType == OptimizationOutput::NodeType::dataMovement) {
      node.direction = output.nodeIdToDataMovementMap[i].direction;
      node.arrayId = output.nodeIdToDataMovementMap[i].arrayId;
    }

    serializableNodes.push_back(node);
  }

  nlohmann::json j;
  j["nodes"] = serializableNodes;
  j["arraysInitiallyAllocatedOnDevice"] = output.arraysInitiallyAllocatedOnDevice;
  std::string s = j.dump(2);
  std::ofstream f(path);
  f << s << std::endl;
}

OptimizationOutput loadOptimizationOutput(const std::string &path) {
  LOG_TRACE_WITH_INFO("Loading optimization plan from %s", path.c_str());

  std::ifstream f(path);
  auto j = nlohmann::json::parse(f);
  auto serializableNodes = j.at("nodes").get<std::vector<SerializableOptimizationOutputNode>>();

  OptimizationOutput output;
  for (const auto &node : serializableNodes) {
    output.nodes.push_back(node.nodeId);
    output.edges[node.nodeId] = node.edges;
    output.nodeIdToNodeTypeMap[node.nodeId] = node.nodeType;
    if (node.nodeType == OptimizationOutput::NodeType::task) {
      output.nodeIdToTaskIdMap[node.nodeId] = node.taskId;
    } else {
      output.nodeIdToDataMovementMap[node.nodeId] = {node.direction, node.arrayId};
    }
  }

  output.arraysInitiallyAllocatedOnDevice = j.at("arraysInitiallyAllocatedOnDevice").get<std::vector<ArrayId>>();

  return output;
}

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

OptimizationOutput Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  if (ConfigurationManager::getConfig().optimization.loadExistingPlan) {
    return loadOptimizationOutput(ConfigurationManager::getConfig().optimization.planPath);
  }

  registerDummyKernelHandles();
  ScopeGuard scopeGuard([]() -> void { cleanUpDummyKernelFuncHandleRegistrations(); });

  auto timeline = getCudaGraphExecutionTimeline(originalGraph);

  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(originalGraph, nodes, edges);

  std::map<cudaGraphNode_t, cudaGraphNode_t> nodeToAnnotationMap;
  mapNodeToAnnotation(originalGraph, edges, nodeToAnnotationMap);

  DisjointSet<cudaGraphNode_t> disjointSet;

  if (ConfigurationManager::getConfig().optimization.mergeConcurrentCudaGraphNodes) {
    mergeConcurrentCudaGraphNodes(timeline, disjointSet, std::numeric_limits<int>::max());
  }

  mergeNodesWithSameAnnotation(nodes, nodeToAnnotationMap, disjointSet);

  std::map<TaskId, OptimizationInput::TaskGroup::DataDependency> taskIdToDataDependencyMap;
  getTaskDataDependencies(nodes, nodeToAnnotationMap, taskIdToDataDependencyMap);

  bool hasOnlyOneStage = true;
  for (auto node : nodes) {
    if (compareKernelNodeFunctionHandle(node, dummyKernelForStageSeparatorHandle)) {
      hasOnlyOneStage = false;
      break;
    }
  }

  std::map<cudaGraphNode_t, int> nodeToStageIndexMap;
  if (!hasOnlyOneStage) {
    mapNodeToStage(originalGraph, edges, nodeToStageIndexMap);
  }

  if (hasOnlyOneStage) {
    auto optimizationInput = constructOptimizationInput(originalGraph, nodes, edges, timeline, disjointSet, nodeToAnnotationMap, taskIdToDataDependencyMap);

    auto optimizationOutput = this->optimize<TwoStepOptimizationStrategy>(optimizationInput);

    if (optimizationOutput.optimal) {
      writeOptimizationOutputToFile(optimizationOutput, ConfigurationManager::getConfig().optimization.planPath);
      return optimizationOutput;
    } else {
      LOG_TRACE_WITH_INFO("Could not find any feasible solution");
      exit(-1);
    }
  } else {
    auto optimizationInputs = constructOptimizationInputsForStages(
      originalGraph, nodes, edges, timeline, disjointSet, nodeToAnnotationMap, taskIdToDataDependencyMap, nodeToStageIndexMap
    );

    std::vector<OptimizationOutput> optimizationOutputs;
    for (int i = 0; i < optimizationInputs.size(); i++) {
      optimizationOutputs.push_back(this->optimize<TwoStepOptimizationStrategy>(optimizationInputs[i]));
      if (optimizationOutputs.rbegin()->optimal == false) {
        LOG_TRACE_WITH_INFO("Could not find any feasible solution for stage %d", i);
        exit(-1);
      }
    }

    auto mergedOptimizationOutput = mergeOptimizationOutputs(optimizationOutputs);
    writeOptimizationOutputToFile(mergedOptimizationOutput, ConfigurationManager::getConfig().optimization.planPath);
    return mergedOptimizationOutput;
  }
}

}  // namespace memopt
