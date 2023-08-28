#pragma once

#include <cuda.h>
#include <cuda_stdint.h>

#include <map>

// Take the address of the node as the ID
typedef uint64_t GraphNodeId;

class TaskManager {
 public:
  static TaskManager *getInstance();
  TaskManager(TaskManager &other) = delete;
  void operator=(const TaskManager &) = delete;

  // Run kernels one by one and record running times.
  // Return a map containing all kernel nodes' running time.
  std::map<GraphNodeId, float> getKernelRunningTimes(cudaGraph_t graph);

 protected:
  TaskManager() = default;
  static TaskManager *instance;

 private:
  cudaStream_t sequentialStream;
  void initializeSequentialExecutionEnvironment();
  void executeNodeSequentially(CUgraphNode node);
  void finalizeSequentialExecutionEnvironment();
};
