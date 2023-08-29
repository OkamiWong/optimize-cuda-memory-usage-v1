#pragma once

#include <utility>
#include <vector>

struct MemoryManager {
  inline static std::vector<std::tuple<void *, size_t>> managedMemorySpaces;
};

template <typename T>
__host__ cudaError_t wrappedCudaMallocManaged(T **devPtr, size_t size) {
  auto err = cudaMallocManaged(devPtr, size);
  MemoryManager::managedMemorySpaces.push_back(std::make_tuple((void *)*devPtr, size));
  return err;
}
