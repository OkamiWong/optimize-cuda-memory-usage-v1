#include <cassert>
#include <memory>
#include <queue>

#include "../profiling/memoryManager.hpp"
#include "../profiling/peakMemoryUsageProfiler.hpp"
#include "../utilities/configurationManager.hpp"
#include "../utilities/constants.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "executor.hpp"

class OptimizedCudaGraphCreator {
 public:
  OptimizedCudaGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph) {}

  void beginCaptureOperation(const std::vector<cudaGraphNode_t> &dependencies) {
    checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
  }

  std::vector<cudaGraphNode_t> endCaptureOperation() {
    checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
    return this->getNewLeafNodesAddedByLastCapture();
  };

  cudaGraphNode_t addEmptyNode(const std::vector<cudaGraphNode_t> &dependencies) {
    cudaGraphNode_t newEmptyNode;
    checkCudaErrors(cudaGraphAddEmptyNode(&newEmptyNode, this->graph, dependencies.data(), dependencies.size()));
    return newEmptyNode;
  }

 private:
  cudaStream_t stream;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> lastDependencies;
  std::map<cudaGraphNode_t, bool> visited;

  std::vector<cudaGraphNode_t> getNewLeafNodesAddedByLastCapture() {
    size_t numNodes;
    checkCudaErrors(cudaGraphGetNodes(this->graph, nullptr, &numNodes));
    auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
    checkCudaErrors(cudaGraphGetNodes(this->graph, nodes.get(), &numNodes));

    size_t numEdges;
    checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
    auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
    auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
    checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

    std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
    for (int i = 0; i < numEdges; i++) {
      hasOutGoingEdge[from[i]] = true;
    }

    std::vector<cudaGraphNode_t> newLeafNodes;
    for (int i = 0; i < numNodes; i++) {
      auto &node = nodes[i];
      if (!visited[node]) {
        visited[node] = true;
        if (!hasOutGoingEdge[node]) {
          newLeafNodes.push_back(node);
        }
      }
    }

    return newLeafNodes;
  }
};

Executor *Executor::instance = nullptr;

Executor *Executor::getInstance() {
  if (instance == nullptr) {
    instance = new Executor();
  }
  return instance;
}

float Executor::executeOptimizedGraph(OptimizationOutput &optimizedGraph, ExecuteRandomTask executeRandomTask) {
  LOG_TRACE_WITH_INFO("Initialize");
  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  auto optimizedCudaGraphCreator = std::make_unique<OptimizedCudaGraphCreator>(stream, graph);

  std::map<int, int> inDegrees;
  for (auto &[u, outEdges] : optimizedGraph.edges) {
    for (auto &v : outEdges) {
      inDegrees[v] += 1;
    }
  }

  std::queue<int> nodesToExecute;
  std::vector<int> rootNodes;
  for (auto &u : optimizedGraph.nodes) {
    if (inDegrees[u] == 0) {
      nodesToExecute.push(u);
      rootNodes.push_back(u);
    }
  }

  int storageDeviceId = ConfigurationManager::getConfig().useNvlink ? Constants::STORAGE_DEVICE_ID : cudaCpuDeviceId;
  cudaMemcpyKind prefetchMemcpyKind = ConfigurationManager::getConfig().useNvlink ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  cudaMemcpyKind offloadMemcpyKind = ConfigurationManager::getConfig().useNvlink ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

  if (ConfigurationManager::getConfig().useNvlink) {
    enablePeerAccessForNvlink(Constants::DEVICE_ID, Constants::STORAGE_DEVICE_ID);
  }

  LOG_TRACE_WITH_INFO("Initialize managed data distribution");
  for (auto ptr : MemoryManager::managedMemoryAddresses) {
    checkCudaErrors(cudaMemPrefetchAsync(
      ptr,
      MemoryManager::managedMemoryAddressToSizeMap[ptr],
      storageDeviceId
    ));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  std::map<void *, void *> addressUpdate;
  std::map<int, std::vector<cudaGraphNode_t>> nodeToDependentNodesMap;

  LOG_TRACE_WITH_INFO("Record nodes to a new CUDA Graph");

  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (auto ptr : optimizedGraph.arraysInitiallyAllocatedOnDevice) {
    auto size = MemoryManager::managedMemoryAddressToSizeMap[ptr];

    void *devicePtr;
    checkCudaErrors(cudaMallocAsync(&devicePtr, size, stream));
    checkCudaErrors(cudaMemcpyAsync(devicePtr, ptr, size, prefetchMemcpyKind, stream));
    addressUpdate[ptr] = devicePtr;
  }
  cudaGraph_t graphForInitialDataDistribution;
  checkCudaErrors(cudaStreamEndCapture(stream, &graphForInitialDataDistribution));

  cudaGraphExec_t graphExecForInitialDataDistribution;
  checkCudaErrors(cudaGraphInstantiate(&graphExecForInitialDataDistribution, graphForInitialDataDistribution, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExecForInitialDataDistribution, stream));
  checkCudaErrors(cudaDeviceSynchronize());

  // Kahn Algorithm
  while (!nodesToExecute.empty()) {
    auto u = nodesToExecute.front();
    nodesToExecute.pop();

    std::vector<cudaGraphNode_t> newLeafNodes;

    auto nodeType = optimizedGraph.nodeIdToNodeTypeMap[u];
    if (nodeType == OptimizationOutput::NodeType::dataMovement) {
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      auto &dataMovement = optimizedGraph.nodeIdToDataMovementMap[u];
      auto dataMovementSize = MemoryManager::managedMemoryAddressToSizeMap[dataMovement.address];
      if (dataMovement.direction == OptimizationOutput::DataMovement::Direction::hostToDevice) {
        void *devicePtr;
        checkCudaErrors(cudaMallocAsync(&devicePtr, dataMovementSize, stream));
        checkCudaErrors(cudaMemcpyAsync(devicePtr, dataMovement.address, dataMovementSize, prefetchMemcpyKind, stream));
        addressUpdate[dataMovement.address] = devicePtr;
      } else {
        void *devicePtr = addressUpdate[dataMovement.address];
        checkCudaErrors(cudaMemcpyAsync(dataMovement.address, devicePtr, dataMovementSize, offloadMemcpyKind, stream));
        checkCudaErrors(cudaFreeAsync(devicePtr, stream));
        addressUpdate.erase(dataMovement.address);
      }
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
    } else if (nodeType == OptimizationOutput::NodeType::task) {
      optimizedCudaGraphCreator->beginCaptureOperation(nodeToDependentNodesMap[u]);
      executeRandomTask(
        optimizedGraph.nodeIdToTaskIdMap[u],
        addressUpdate,
        stream
      );
      newLeafNodes = optimizedCudaGraphCreator->endCaptureOperation();
    } else if (nodeType == OptimizationOutput::NodeType::empty) {
      newLeafNodes.push_back(
        optimizedCudaGraphCreator->addEmptyNode(nodeToDependentNodesMap[u])
      );
    } else {
      LOG_TRACE_WITH_INFO("Unsupported node type: %d", nodeType);
      exit(-1);
    }

    for (auto &v : optimizedGraph.edges[u]) {
      inDegrees[v]--;
      nodeToDependentNodesMap[v].insert(
        nodeToDependentNodesMap[v].begin(),
        newLeafNodes.begin(),
        newLeafNodes.end()
      );

      if (inDegrees[v] == 0) {
        nodesToExecute.push(v);
      }
    }
  }

  LOG_TRACE_WITH_INFO("Printing the new CUDA Graph to newGraph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "newGraph.dot", 0));

  LOG_TRACE_WITH_INFO("Execute the new CUDA Graph");
  PeakMemoryUsageProfiler peakMemoryUsageProfiler;
  CudaEventClock cudaEventClock;
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  if (ConfigurationManager::getConfig().measurePeakMemoryUsage) {
    peakMemoryUsageProfiler.start();
  }

  cudaEventClock.start();
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  cudaEventClock.end();
  checkCudaErrors(cudaDeviceSynchronize());

  if (ConfigurationManager::getConfig().measurePeakMemoryUsage) {
    const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
    LOG_TRACE_WITH_INFO(
      "Peak memory usage (MiB): %.2f\n",
      static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
    );
  }

  LOG_TRACE_WITH_INFO("Clean up");
  for (auto &[oldAddr, newAddr] : addressUpdate) {
    checkCudaErrors(cudaMemcpy(oldAddr, newAddr, MemoryManager::managedMemoryAddressToSizeMap[oldAddr], offloadMemcpyKind));
    checkCudaErrors(cudaFree(newAddr));
  }
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaGraphExecDestroy(graphExecForInitialDataDistribution));
  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphDestroy(graphForInitialDataDistribution));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaStreamDestroy(stream));

  if (ConfigurationManager::getConfig().useNvlink) {
    disablePeerAccessForNvlink(Constants::DEVICE_ID, Constants::STORAGE_DEVICE_ID);
  }

  return cudaEventClock.getTimeInSeconds();
}
