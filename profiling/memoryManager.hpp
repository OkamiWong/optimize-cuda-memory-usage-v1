#pragma once

#include <initializer_list>
#include <map>
#include <set>

struct MemoryManager {
  inline static int managedMemoryAddressCount = 0;
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  inline static std::map<void *, int> managedMemoryAddressToIndexMap;
};

template <typename T>
__host__ void registerManagedMemoryAddress(T *devPtr, size_t size) {
  MemoryManager::managedMemoryAddressToSizeMap[static_cast<void *>(devPtr)] = size;
  MemoryManager::managedMemoryAddressToIndexMap[static_cast<void *>(devPtr)] = MemoryManager::managedMemoryAddressCount++;
}

template <typename T>
__host__ cudaError_t wrappedCudaMallocManaged(T **devPtr, size_t size) {
  auto err = cudaMallocManaged(devPtr, size);
  registerManagedMemoryAddress(*devPtr, size);
  return err;
}
