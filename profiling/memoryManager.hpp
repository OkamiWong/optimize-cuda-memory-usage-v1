#pragma once

#include <cassert>
#include <initializer_list>
#include <map>
#include <set>
#include <vector>

#include "../utilities/types.hpp"

struct MemoryManager {
  inline static std::vector<void *> managedMemoryAddresses;
  inline static std::map<void *, ArrayId> managedMemoryAddressToIndexMap;
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  inline static std::set<void *> applicationInputs, applicationOutputs;
};

template <typename T>
void registerManagedMemoryAddress(T *devPtr, size_t size) {
  auto ptr = static_cast<void *>(devPtr);
  if (MemoryManager::managedMemoryAddressToIndexMap.find(ptr) == MemoryManager::managedMemoryAddressToIndexMap.end()) {
    MemoryManager::managedMemoryAddresses.push_back(ptr);
    MemoryManager::managedMemoryAddressToIndexMap[ptr] = MemoryManager::managedMemoryAddresses.size() - 1;
    MemoryManager::managedMemoryAddressToSizeMap[ptr] = size;
  }
}

template <typename T>
void registerApplicationInput(T *devPtr) {
  MemoryManager::applicationInputs.insert(static_cast<void *>(devPtr));
}

template <typename T>
void registerApplicationOutput(T *devPtr) {
  MemoryManager::applicationOutputs.insert(static_cast<void *>(devPtr));
}

void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap);
