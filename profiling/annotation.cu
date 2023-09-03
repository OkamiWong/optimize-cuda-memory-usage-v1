#include <cstring>

#include "annotation.hpp"

__global__ void dummyKernelForAnnotation(KernelIO io) {
  return;
}

__host__ void annotateNextKernel(
  std::initializer_list<void *> inputs,
  std::initializer_list<void *> outputs,
  cudaStream_t stream
) {
  KernelIO io;
  memset(io.inputs, 0, KernelIO::MAX_NUM_PTR * sizeof(void *));
  memset(io.outputs, 0, KernelIO::MAX_NUM_PTR * sizeof(void *));
  memcpy(io.inputs, std::data(inputs), inputs.size() * sizeof(void *));
  memcpy(io.outputs, std::data(outputs), outputs.size() * sizeof(void *));
  dummyKernelForAnnotation<<<1, 1, 0, stream>>>(io);
}
