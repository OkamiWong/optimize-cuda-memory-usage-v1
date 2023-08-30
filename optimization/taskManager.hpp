#pragma once

#include <cuda.h>
#include <cuda_stdint.h>

#include <map>

class TaskManager {
 public:
  static TaskManager *getInstance();
  TaskManager(TaskManager &other) = delete;
  void operator=(const TaskManager &) = delete;

  // Take the address of the node as the ID
  typedef uint64_t GraphNodeId;

  // Run kernels one by one and record running times.
  // Return a map containing all kernel nodes' running time.
  std::map<GraphNodeId, float> getKernelRunningTimes(cudaGraph_t graph);

 protected:
  TaskManager() = default;
  static TaskManager *instance;

 private:
  cudaStream_t sequentialStream;
  CUfunction dummyKernelHandle;

  void initializeSequentialExecutionEnvironment();

  // Return false when the node is a dummy node which is not
  // going to be executed.
  bool executeNodeSequentially(CUgraphNode node);

  void finalizeSequentialExecutionEnvironment();
};
