#include <algorithm>
#include <cstring>
#include <functional>

#include "annotation.hpp"
#include "memoryManager.hpp"

namespace memopt {

__global__ void dummyKernelForAnnotation(TaskAnnotation taskAnnotation) {
  return;
}

__host__ void annotateNextTask(
  TaskId taskId,
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  cudaStream_t stream
) {
  TaskAnnotation taskAnnotation;
  taskAnnotation.taskId = taskId;

  auto isManagedArray = [&](void *arr) {
    return MemoryManager::managedMemoryAddressToIndexMap.count(arr) > 0;
  };

  // Remove arrays that are not managed
  auto inputsNewEnd = std::remove_if(inputs.begin(), inputs.end(), std::not_fn(isManagedArray));
  auto outputsNewEnd = std::remove_if(outputs.begin(), outputs.end(), std::not_fn(isManagedArray));
  inputs.erase(inputsNewEnd, inputs.end());
  outputs.erase(outputsNewEnd, outputs.end());

  memset(taskAnnotation.inputs, 0, TaskAnnotation::MAX_NUM_PTR * sizeof(void *));
  memset(taskAnnotation.outputs, 0, TaskAnnotation::MAX_NUM_PTR * sizeof(void *));
  memcpy(taskAnnotation.inputs, std::data(inputs), inputs.size() * sizeof(void *));
  memcpy(taskAnnotation.outputs, std::data(outputs), outputs.size() * sizeof(void *));

  dummyKernelForAnnotation<<<1, 1, 0, stream>>>(taskAnnotation);
}

__host__ void annotateNextTask(
  TaskId taskId,
  cudaStream_t stream
) {
  annotateNextTask(taskId, {}, {}, stream);
}

__global__ void dummyKernelForStageSeparator() {
  return;
}

__host__ void endStage(
  cudaStream_t stream
) {
  dummyKernelForStageSeparator<<<1, 1, 0, stream>>>();
}

}  // namespace memopt
