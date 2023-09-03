#pragma once

#include <initializer_list>
#include <set>
#include <utility>
#include <vector>

struct MemoryManager {
  typedef std::tuple<void *, size_t> ArrayInfo;
  inline static std::vector<ArrayInfo> managedMemorySpaces;
  inline static std::set<void *> managedMemorySpacesInitiallyOnDevice;
};

template <typename T>
__host__ cudaError_t wrappedCudaMallocManaged(T **devPtr, size_t size) {
  auto err = cudaMallocManaged(devPtr, size);
  MemoryManager::managedMemorySpaces.push_back(std::make_tuple((void *)*devPtr, size));
  return err;
}

void markMemorySpaceInitiallyOnDevice(void *devPtr) {
  MemoryManager::managedMemorySpacesInitiallyOnDevice.insert(devPtr);
}

void markMemorySpaceInitiallyOnDevice(std::initializer_list<void *> devPtrs) {
  for (auto &devPtr : devPtrs) {
    markMemorySpaceInitiallyOnDevice(devPtr);
  }
}
