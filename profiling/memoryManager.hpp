#pragma once

#include <initializer_list>
#include <map>
#include <set>

struct MemoryManager {
  typedef std::tuple<void *, size_t> ArrayInfo;
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  inline static std::set<void *> managedMemorySpacesInitiallyOnDevice;
};

template <typename T>
__host__ cudaError_t wrappedCudaMallocManaged(T **devPtr, size_t size) {
  auto err = cudaMallocManaged(devPtr, size);
  MemoryManager::managedMemoryAddressToSizeMap[(void *)*devPtr] = size;
  return err;
}

void markMemorySpaceInitiallyOnDevice(void *devPtr);

void markMemorySpaceInitiallyOnDevice(std::initializer_list<void *> devPtrs);
