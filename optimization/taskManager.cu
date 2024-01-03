#include <cuda.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "../include/bipartiteGraphMaximumMatching.hpp"
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

// Take the only root node in the graph as the dummy kernel.
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

void TaskManager::queueCudaKernelToStream(cudaGraphNode_t node, cudaStream_t stream) {
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
    if (params.func == this->dummyKernelHandle) return;
    checkCudaErrors(cuLaunchKernelEx(
      &config,
      params.func,
      params.kernelParams,
      params.extra
    ));
  } else if (params.kern != nullptr) {
    auto func = reinterpret_cast<CUfunction>(params.kern);
    if (func == this->dummyKernelHandle) return;
    checkCudaErrors(cuLaunchKernelEx(
      &config,
      func,
      params.kernelParams,
      params.extra
    ));
  } else {
    LOG_TRACE_WITH_INFO("Currently only support params.func != NULL or params.kernel != NULL");
    exit(-1);
  }
}

void TaskManager::queueCudaMemsetToStream(cudaGraphNode_t node, cudaStream_t stream) {
  CUDA_MEMSET_NODE_PARAMS params;
  checkCudaErrors(cuGraphMemsetNodeGetParams(node, &params));
  if (params.elementSize == 1) {
    cuMemsetD2D8Async(params.dst, params.pitch, params.value, params.width, params.height, stream);
  } else if (params.elementSize == 2) {
    cuMemsetD2D16Async(params.dst, params.pitch, params.value, params.width, params.height, stream);
  } else if (params.elementSize == 4) {
    cuMemsetD2D32Async(params.dst, params.pitch, params.value, params.width, params.height, stream);
  }
}

void TaskManager::queueCudaNodeToStream(cudaGraphNode_t node, cudaStream_t stream) {
  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(node, &nodeType));
  if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
    this->queueCudaKernelToStream(node, stream);
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMSET) {
    this->queueCudaMemsetToStream(node, stream);
  } else {
    LOG_TRACE_WITH_INFO("Currently only support executing kernel or memset");
    exit(-1);
  }
}

std::map<CustomGraph::NodeId, TaskManager::StreamId> TaskManager::getStreamAssignment(
  CustomGraph &optimizedGraph
) {
  augment_path augmentPathAlgorithm(optimizedGraph.nodes.size(), optimizedGraph.nodes.size());
  for (auto &[u, edges] : optimizedGraph.edges) {
    for (auto v : edges) {
      augmentPathAlgorithm.add(u, v);
    }
  }
  augmentPathAlgorithm.solve();

  std::map<CustomGraph::NodeId, TaskManager::StreamId> nodeIdToStreamIdMap;
  TaskManager::StreamId nextStreamId = 0;
  for (auto u : optimizedGraph.nodes) {
    if (nodeIdToStreamIdMap.count(u) == 0) {
      auto uStreamId = nextStreamId++;
      nodeIdToStreamIdMap[u] = uStreamId;
      for (auto v = augmentPathAlgorithm.pa[u]; v != -1; v = augmentPathAlgorithm.pa[v]) {
        nodeIdToStreamIdMap[v] = uStreamId;
      }
    }
  }

  return nodeIdToStreamIdMap;
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
          ? Constants::DEVICE_ID
          : cudaCpuDeviceId,
        uStream
      ));
    } else if (nodeType == CustomGraph::NodeType::kernel) {
      this->queueCudaNodeToStream(optimizedGraph.nodeIdToCuGraphNodeMap[u], uStream);
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
