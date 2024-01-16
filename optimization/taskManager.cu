#include <cuda.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
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
  config.attrs = nullptr;
  config.numAttrs = 0;

  if (params.extra != nullptr) {
    LOG_TRACE_WITH_INFO("Currently only support params.extra == nullptr");
    exit(-1);
  }

  // Replace recorded memory addresses with actual ones
  void **modifiedKernelParams = nullptr;
  void **kernelParams = params.kernelParams;
  std::vector<void *> modifiedKernelParamsContainer;
  if (kernelParams != nullptr) {
    while (true) {
      auto addr = static_cast<void **>(kernelParams[0])[0];
      if (this->recordedAddressToActualAddressMap.count(addr) != 0) {
        puts("replace addr");
        modifiedKernelParamsContainer.push_back(&(this->recordedAddressToActualAddressMap[addr]));
      } else {
        modifiedKernelParamsContainer.push_back(kernelParams[0]);
      }

      int offset = static_cast<char *>(kernelParams[1]) - static_cast<char *>(kernelParams[0]);
      printf("%d\n", offset);
      if (offset != 2 && offset != 4 && offset != 8 && offset != 16 && offset != 32 && offset != 64 && offset != 56) {
        break;
      }

      kernelParams++;
    }
    modifiedKernelParams = modifiedKernelParamsContainer.data();
  }
  puts("kernelParam updated\n");

  if (params.func != nullptr) {
    if (params.func == this->dummyKernelHandle) return;
    checkCudaErrors(cuLaunchKernelEx(
      &config,
      params.func,
      modifiedKernelParams,
      nullptr
    ));
  } else if (params.kern != nullptr) {
    auto func = reinterpret_cast<CUfunction>(params.kern);
    if (func == this->dummyKernelHandle) return;
    checkCudaErrors(cuLaunchKernelEx(
      &config,
      func,
      modifiedKernelParams,
      nullptr
    ));
  } else {
    LOG_TRACE_WITH_INFO("Currently only support params.func != nullptr or params.kernel != nullptr");
    exit(-1);
  }
  cuMemCreate();
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
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
    CUDA_MEM_ALLOC_NODE_PARAMS params;
    checkCudaErrors(cuGraphMemAllocNodeGetParams(node, &params));
    void *ptr;
    checkCudaErrors(cudaMallocAsync(&ptr, params.bytesize, stream));
    this->recordedAddressToActualAddressMap[reinterpret_cast<void *>(params.dptr)] = ptr;
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEM_FREE) {
    CUdeviceptr dptr;
    checkCudaErrors(cuGraphMemFreeNodeGetParams(node, &dptr));
    checkCudaErrors(cudaFreeAsync(
      this->recordedAddressToActualAddressMap[reinterpret_cast<void *>(dptr)],
      stream
    ));
  } else {
    LOG_TRACE_WITH_INFO("Currently only support executing kernel, memset, mem alloc, and mem free");
    exit(-1);
  }
}

std::unordered_map<CustomGraph::NodeId, TaskManager::StreamId> TaskManager::getStreamAssignment(
  CustomGraph &optimizedGraph
) {
  augment_path augmentPathAlgorithm(optimizedGraph.nodes.size(), optimizedGraph.nodes.size());
  for (auto &[u, edges] : optimizedGraph.edges) {
    for (auto v : edges) {
      augmentPathAlgorithm.add(u, v);
    }
  }
  augmentPathAlgorithm.solve();

  std::unordered_map<CustomGraph::NodeId, TaskManager::StreamId> nodeIdToStreamIdMap;
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
  this->recordedAddressToActualAddressMap.clear();

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

  LOG_TRACE_WITH_INFO("All nodes are queued to streams");

  for (;;) {
    sleep(1);
    std::vector<TaskManager::StreamId> unfinishedStreams;
    for (auto &[streamId, stream] : streamIdToCudaStreamMap) {
      auto result = cudaStreamQuery(stream);
      if (result != cudaSuccess) {
        if (result == cudaErrorNotReady) {
          unfinishedStreams.push_back(streamId);
        } else {
          checkCudaErrors(result);
        }
      }
    }

    if (unfinishedStreams.size() == 0) {
      break;
    }

    printf("Unfinished streams: ");
    for (auto id : unfinishedStreams) {
      printf("%d, ", id);
    }
    printf("\n\n");
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
