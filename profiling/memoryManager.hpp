#pragma once

#include <initializer_list>
#include <map>
#include <set>

struct MemoryManager {
  inline static int managedMemoryAddressCount = 0;
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  inline static std::map<void *, int> managedMemoryAddressToIndexMap;
  inline static std::set<void *> applicationInputs, applicationOutputs;
};

template <typename T>
__host__ void registerManagedMemoryAddress(T *devPtr, size_t size) {
  auto ptr = static_cast<void *>(devPtr);
  if (MemoryManager::managedMemoryAddressToIndexMap.find(ptr) == MemoryManager::managedMemoryAddressToIndexMap.end()) {
    MemoryManager::managedMemoryAddressToIndexMap[ptr] = MemoryManager::managedMemoryAddressCount++;
    MemoryManager::managedMemoryAddressToSizeMap[ptr] = size;
  }
}

template <typename T>
__host__ cudaError_t wrappedCudaMallocManaged(T **devPtr, size_t size) {
  auto err = cudaMallocManaged(devPtr, size);
  registerManagedMemoryAddress(*devPtr, size);
  return err;
}

template <typename T>
__host__ void registerApplicationInput(T *devPtr) {
  MemoryManager::applicationInputs.insert(static_cast<void *>(devPtr));
}

template <typename T>
__host__ void registerApplicationOutput(T *devPtr) {
  MemoryManager::applicationOutputs.insert(static_cast<void *>(devPtr));
}
