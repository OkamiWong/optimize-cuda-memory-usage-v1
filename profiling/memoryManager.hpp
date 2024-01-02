#pragma once

#include <initializer_list>
#include <map>
#include <set>

struct MemoryManager {
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  inline static std::set<void *> managedMemorySpacesInitiallyOnDevice;
};

template <typename T>
__host__ void registerManagedMemoryAddress(T *devPtr, size_t size) {
  MemoryManager::managedMemoryAddressToSizeMap[static_cast<void *>(devPtr)] = size;
}

template <typename T>
__host__ cudaError_t wrappedCudaMallocManaged(T **devPtr, size_t size) {
  auto err = cudaMallocManaged(devPtr, size);
  registerManagedMemoryAddress(*devPtr, size);
  return err;
}

void markMemorySpaceInitiallyOnDevice(void *devPtr);

void markMemorySpaceInitiallyOnDevice(std::initializer_list<void *> devPtrs);
