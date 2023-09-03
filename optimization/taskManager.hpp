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

  // Take the only root node in the graph as the dummy kernel.
  void registerDummyKernelHandle(cudaGraph_t graph);

  CUfunction getDummyKernelHandle();

  // Run kernels one by one and record durations.
  // Return a map containing the duration of each kernel.
  Optimizer::CuGraphNodeToKernelDurationMap getCuGraphNodeToKernelDurationMap(cudaGraph_t graph);

  typedef int StreamId;

  void executeOptimizedGraph(CustomGraph &optimizedGraph);

 protected:
  TaskManager() = default;
  static TaskManager *instance;

 private:
  CUfunction dummyKernelHandle = nullptr;

  cudaStream_t sequentialStream;

  void queueKernelToStream(CUgraphNode node, cudaStream_t stream);

  void initializeSequentialExecutionEnvironment();

  // Return false when the node is a dummy node which is not
  // going to be executed.
  bool executeNodeSequentially(CUgraphNode node);

  void finalizeSequentialExecutionEnvironment();

  // Calculate the stream assignment by bipartite matching
  std::map<CustomGraph::NodeId, StreamId> getStreamAssignment(CustomGraph &optimizedGraph);
};
