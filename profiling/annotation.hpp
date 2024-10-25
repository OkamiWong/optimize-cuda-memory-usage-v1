#pragma once

#include <vector>

#include "../utilities/types.hpp"

namespace memopt {

struct TaskAnnotation {
  // Before CUDA 12.1, the size of the parameters passed to a kernel is
  // limited to 4096 bytes. CUDA 12.1 increases this limit to 32764 bytes.
  static constexpr size_t MAX_NUM_PTR = (4096 - sizeof(TaskId)) / 8 / 2;

  TaskId taskId;

  void *inputs[MAX_NUM_PTR];
  void *outputs[MAX_NUM_PTR];
};

__global__ void dummyKernelForAnnotation(TaskAnnotation taskAnnotation, bool inferDataDependency);

__host__ void annotateNextTask(
  TaskId taskId,
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  cudaStream_t stream
);

__host__ void annotateNextTask(
  TaskId taskId,
  cudaStream_t stream
);

__global__ void dummyKernelForStageSeparator();

__host__ void endStage(
  cudaStream_t stream
);

}  // namespace memopt
