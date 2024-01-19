#pragma once

#include <initializer_list>
#include <iterator>

struct TaskAnnotation {
  // Before CUDA 12.1, the size of the parameters passed to a kernel is
  // limited to 4096 bytes. CUDA 12.1 increases this limit to 32764 bytes.
  static constexpr size_t MAX_NUM_PTR = 4096 / 8 / 2;

  int taskId;

  void *inputs[MAX_NUM_PTR];
  void *outputs[MAX_NUM_PTR];
};

__global__ void dummyKernelForAnnotation(TaskAnnotation taskAnnotation);

__host__ void annotateNextKernel(
  int taskId,
  std::initializer_list<void *> inputs,
  std::initializer_list<void *> outputs,
  cudaStream_t stream
);
