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

void TaskManager::initializeSequentialExecutionEnvironment() {
  checkCudaErrors(cudaStreamCreate(&(this->sequentialStream)));
}

void TaskManager::finalizeSequentialExecutionEnvironment() {
  checkCudaErrors(cudaStreamDestroy(this->sequentialStream));
}

void TaskManager::executeNodeSequentially(CUgraphNode node) {
  CUgraphNodeType nodeType;
  checkCudaErrors(cuGraphNodeGetType(node, &nodeType));
  if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
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
    config.hStream = this->sequentialStream;

    // Currently kernel attributes are ignored
    config.attrs = NULL;
    config.numAttrs = 0;

    // TODO: translate pointers in the parameters of the kernel

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
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEM_ALLOC) {
    CUDA_MEM_ALLOC_NODE_PARAMS params;
    checkCudaErrors(cuGraphMemAllocNodeGetParams(node, &params));
    void *ptr;
    checkCudaErrors(cudaMallocAsync(&ptr, params.bytesize, this->sequentialStream));
    this->actualAddressInSequentialExecution[params.dptr] = ptr;
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEM_FREE) {
    CUdeviceptr dptr;
    checkCudaErrors(cuGraphMemFreeNodeGetParams(node, &dptr));
    checkCudaErrors(cudaFreeAsync(this->actualAddressInSequentialExecution[dptr], this->sequentialStream));
  } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMSET) {
    CUDA_MEMSET_NODE_PARAMS params;
    checkCudaErrors(cuGraphMemsetNodeGetParams(node, &params));

    uint32_t value = 0;
    if (params.elementSize == 1) {
      uint8_t v = params.value;
      for (int i = 0; i < 4; i++) {
        value += v;
        value <<= 8;
      }
    }
    if (params.elementSize == 2) {
      uint16_t v = params.value;
      for (int i = 0; i < 2; i++) {
        value += v;
        value <<= 16;
      }
    }
    if (params.elementSize == 4) {
      value = params.value;
    }

    checkCudaErrors(cudaMemset2DAsync(
      this->actualAddressInSequentialExecution[params.dst],
      params.pitch,
      value,
      params.width,
      params.height,
      this->sequentialStream
    ));
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

  std::queue<CUgraphNode> nodesToExecute;
  for (auto &u : nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
    }
  }

  this->initializeSequentialExecutionEnvironment();

  // Kahn Algorithm
  std::map<GraphNodeId, float> kernelRunningTimes;
  CudaEventClock clock;
  while (!nodesToExecute.empty()) {
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    clock.start(this->sequentialStream);
    this->executeNodeSequentially(u);
    clock.end(this->sequentialStream);
    checkCudaErrors(cudaStreamSynchronize(this->sequentialStream));
    kernelRunningTimes[reinterpret_cast<GraphNodeId>(u)] = clock.getTimeInSeconds();

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
