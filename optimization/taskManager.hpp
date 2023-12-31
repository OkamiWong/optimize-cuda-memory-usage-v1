#pragma once

#include <cuda.h>

#include <map>

#include "customGraph.hpp"
#include "optimizer.hpp"

class TaskManager {
 public:
  static TaskManager *getInstance();
  TaskManager(TaskManager &other) = delete;
  void operator=(const TaskManager &) = delete;

  typedef int StreamId;

  void registerDummyKernelHandle(cudaGraph_t graph);
  CUfunction getDummyKernelHandle();
  void executeOptimizedGraph(CustomGraph &optimizedGraph);

 protected:
  TaskManager() = default;
  static TaskManager *instance;

 private:
  CUfunction dummyKernelHandle = nullptr;

  cudaStream_t sequentialStream;

  void queueCudaNodeToStream(CUgraphNode node, cudaStream_t stream);
  void queueCudaKernelToStream(CUgraphNode node, cudaStream_t stream);
  void queueCudaMemsetToStream(CUgraphNode node, cudaStream_t stream);

  // Calculate the stream assignment by bipartite matching
  std::map<CustomGraph::NodeId, StreamId> getStreamAssignment(CustomGraph &optimizedGraph);
};
