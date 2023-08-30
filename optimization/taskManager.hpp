#pragma once

#include <cuda.h>
#include <cuda_stdint.h>

#include <map>

#include "customGraph.hpp"

class TaskManager {
 public:
  static TaskManager *getInstance();
  TaskManager(TaskManager &other) = delete;
  void operator=(const TaskManager &) = delete;

  // Take the only root node in the graph as the dummy kernel.
  void registerDummyKernelHandle(cudaGraph_t graph);

  // Run kernels one by one and record durations.
  // Return a map containing all kernel nodes' running time.
  std::map<CUgraphNode, float> getKernelRunningTimes(cudaGraph_t graph);

  void executeOptimizedGraph(const CustomGraph &optimizedGraph);

 protected:
  TaskManager() = default;
  static TaskManager *instance;

 private:
  CUfunction dummyKernelHandle;

  cudaStream_t sequentialStream;

  void initializeSequentialExecutionEnvironment();

  // Return false when the node is a dummy node which is not
  // going to be executed.
  bool executeNodeSequentially(CUgraphNode node);

  void finalizeSequentialExecutionEnvironment();
};
