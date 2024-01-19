#include <cstring>

#include "annotation.hpp"

__global__ void dummyKernelForAnnotation(TaskAnnotation io) {
  return;
}

__host__ void annotateNextKernel(
  int taskId,
  std::initializer_list<void *> inputs,
  std::initializer_list<void *> outputs,
  cudaStream_t stream
) {
  TaskAnnotation taskAnnotation;
  taskAnnotation.taskId = taskId;

  memset(taskAnnotation.inputs, 0, TaskAnnotation::MAX_NUM_PTR * sizeof(void *));
  memset(taskAnnotation.outputs, 0, TaskAnnotation::MAX_NUM_PTR * sizeof(void *));
  memcpy(taskAnnotation.inputs, std::data(inputs), inputs.size() * sizeof(void *));
  memcpy(taskAnnotation.outputs, std::data(outputs), outputs.size() * sizeof(void *));

  dummyKernelForAnnotation<<<1, 1, 0, stream>>>(taskAnnotation);
}
