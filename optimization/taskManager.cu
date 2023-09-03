#include <cuda.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "optimizer.hpp"
#include "taskManager.hpp"

TaskManager *TaskManager::instance = nullptr;

TaskManager *TaskManager::getInstance() {
  if (instance == nullptr) {
    instance = new TaskManager();
  }
  return instance;
}

void TaskManager::registerDummyKernelHandle(cudaGraph_t graph) {
  auto rootNode = getRootNode(graph);

  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(rootNode, &nodeType));
  assert(nodeType == CU_GRAPH_NODE_TYPE_KERNEL);

  CUDA_KERNEL_NODE_PARAMS rootNodeParams;
  checkCudaErrors(cuGraphKernelNodeGetParams(rootNode, &rootNodeParams));
  this->dummyKernelHandle = rootNodeParams.func;
}

CUfunction TaskManager::getDummyKernelHandle() {
  assert(this->dummyKernelHandle != nullptr);
  return this->dummyKernelHandle;
}

void TaskManager::initializeSequentialExecutionEnvironment() {
  checkCudaErrors(cudaStreamCreate(&(this->sequentialStream)));
}

void TaskManager::finalizeSequentialExecutionEnvironment() {
  checkCudaErrors(cudaStreamDestroy(this->sequentialStream));
}

void TaskManager::queueKernelToStream(CUgraphNode node, cudaStream_t stream) {
  CUDA_KERNEL_NODE_PARAMS params;
  checkCudaErrors(cuGraphKernelNodeGetParams(node, &params));

  CUlaunchConfig config;
  config.gridDimX = params.gridDimX;
  config.gridDimY = params.gridDimY;
  config.gridDimZ = params.gridDimZ;
  config.blockDimX = params.blockDimX;
  config.blockDimY = params.blockDimY;
  config.blockDimZ = params.blockDimZ;
  config.sharedMemBytes = params.sharedMemBytes;
  config.hStream = stream;

  // Currently kernel attributes are ignored
  config.attrs = NULL;
  config.numAttrs = 0;

  if (params.func != nullptr) {
    checkCudaErrors(cuLaunchKernelEx(
      &config,
      params.func,
      params.kernelParams,
      params.extra
    ));
  } else if (params.kern != nullptr) {
    checkCudaErrors(cuLaunchKernelEx(
      &config,
      reinterpret_cast<CUfunction>(params.kern),
      params.kernelParams,
      params.extra
    ));
  } else {
    LOG_TRACE_WITH_INFO("Currently only support params.func != NULL or params.kernel != NULL");
    exit(-1);
  }
}

bool TaskManager::executeNodeSequentially(CUgraphNode node) {
  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(node, &nodeType));
  if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
    CUDA_KERNEL_NODE_PARAMS params;
    checkCudaErrors(cuGraphKernelNodeGetParams(node, &params));

    if (params.func == this->dummyKernelHandle) {
      return false;
    }

    this->queueKernelToStream(node, this->sequentialStream);
  } else {
    LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
    exit(-1);
  }

  return true;
}

Optimizer::CuGraphNodeToKernelDurationMap TaskManager::getCuGraphNodeToKernelDurationMap(cudaGraph_t graph) {
  std::vector<CUgraphNode> nodes;
  std::map<CUgraphNode, std::vector<CUgraphNode>> edges;
  extractGraphNodesAndEdges(graph, nodes, edges);

  std::map<CUgraphNode, int> inDegrees;
  for (auto &[u, outEdges] : edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  std::queue<CUgraphNode> nodesToExecute;
  for (auto &u : nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
    }
  }

  this->initializeSequentialExecutionEnvironment();

  // Kahn Algorithm
  Optimizer::CuGraphNodeToKernelDurationMap kernelRunningTimes;
  CudaEventClock clock;
  while (!nodesToExecute.empty()) {
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    clock.start(this->sequentialStream);
    auto isExecuted = this->executeNodeSequentially(u);
    clock.end(this->sequentialStream);
    checkCudaErrors(cudaStreamSynchronize(this->sequentialStream));
    if (isExecuted) {
      kernelRunningTimes[u] = clock.getTimeInSeconds();
    }

    for (auto &v : edges[u]) {
      inDegrees[v]--;
      if (inDegrees[v] == 0) {
        nodesToExecute.push(v);
      }
    }
  }

  this->finalizeSequentialExecutionEnvironment();

  return kernelRunningTimes;
}

std::map<CustomGraph::NodeId, TaskManager::StreamId> TaskManager::getStreamAssignment(
  CustomGraph &optimizedGraph
) {
  // TODO
}

void TaskManager::executeOptimizedGraph(CustomGraph &optimizedGraph) {
  // Initialization
  auto nodeIdToStreamIdMap = this->getStreamAssignment(optimizedGraph);
  std::map<TaskManager::StreamId, cudaStream_t> streamIdToCudaStreamMap;
  for (auto &[nodeId, streamId] : nodeIdToStreamIdMap) {
    if (streamIdToCudaStreamMap.count(streamId) == 0) {
      cudaStream_t s;
      checkCudaErrors(cudaStreamCreate(&s));
      streamIdToCudaStreamMap[streamId] = s;
    }
  }

  std::map<CustomGraph::NodeId, int> inDegrees;
  for (auto &[u, outEdges] : optimizedGraph.edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  std::queue<CustomGraph::NodeId> nodesToExecute;
  for (auto &u : optimizedGraph.nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
    }
  }

  std::vector<cudaEvent_t> createdCudaEvents;

  // Launch nodes to assigned streams based on Kahn Algorithm
  while (!nodesToExecute.empty()) {
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    // Execute node
    auto uStream = streamIdToCudaStreamMap[nodeIdToStreamIdMap[u]];
    auto nodeType = optimizedGraph.nodeIdToNodeTypeMap[u];
    if (nodeType == CustomGraph::NodeType::dataMovement) {
      auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap[u];
      checkCudaErrors(cudaMemPrefetchAsync(
        dataMovement.address,
        dataMovement.size,
        dataMovement.direction == CustomGraph::DataMovement::Direction::hostToDevice
          ? CudaConstants::DEVICE_ID
          : cudaCpuDeviceId,
        uStream
      ));
    } else if (nodeType == CustomGraph::NodeType::kernel) {
      this->queueKernelToStream(optimizedGraph.nodeIdToCuGraphNodeMap[u], uStream);
    } else if (nodeType == CustomGraph::NodeType::empty) {
      // Pass
    } else {
      LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
      exit(-1);
    }

    cudaEvent_t e = nullptr;
    for (auto &v : optimizedGraph.edges[u]) {
      inDegrees[v]--;
      if (inDegrees[v] == 0) {
        nodesToExecute.push(v);
      }

      auto vStream = streamIdToCudaStreamMap[nodeIdToStreamIdMap[v]];
      if (uStream != vStream) {
        if (e == nullptr) {
          checkCudaErrors(cudaEventCreate(&e));
          createdCudaEvents.push_back(e);
          checkCudaErrors(cudaEventRecord(e, uStream));
        }
        checkCudaErrors(cudaStreamWaitEvent(vStream, e));
      }
    }
  }

  // Clean-up
  checkCudaErrors(cudaDeviceSynchronize());
  for (auto &[streamId, s] : streamIdToCudaStreamMap) {
    checkCudaErrors(cudaStreamDestroy(s));
  }
  for (auto &e : createdCudaEvents) {
    checkCudaErrors(cudaEventDestroy(e));
  }
}
