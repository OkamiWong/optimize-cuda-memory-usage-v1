#pragma once

#include <cassert>
#include <initializer_list>
#include <map>
#include <set>
#include <vector>

#include "../utilities/configurationManager.hpp"
#include "../utilities/types.hpp"

namespace memopt {

struct MemoryManager {
  inline static std::vector<void *> managedMemoryAddresses;
  inline static std::map<void *, ArrayId> managedMemoryAddressToIndexMap;
  inline static std::map<void *, size_t> managedMemoryAddressToSizeMap;
  inline static std::set<void *> applicationInputs, applicationOutputs;
};

template <typename T>
void registerManagedMemoryAddress(T *devPtr, size_t size) {
  if (size < ConfigurationManager::getConfig().optimization.minManagedArraySize) {
    return;
  }

  auto ptr = static_cast<void *>(devPtr);
  if (MemoryManager::managedMemoryAddressToIndexMap.count(ptr) == 0) {
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

inline void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap) {
  auto oldManagedMemoryAddressToSizeMap = MemoryManager::managedMemoryAddressToSizeMap;

  MemoryManager::managedMemoryAddressToSizeMap.clear();
  MemoryManager::managedMemoryAddressToIndexMap.clear();

  for (int i = 0; i < MemoryManager::managedMemoryAddresses.size(); i++) {
    assert(oldAddressToNewAddressMap.count(MemoryManager::managedMemoryAddresses[i]) == 1);

    const auto newAddr = oldAddressToNewAddressMap.at(MemoryManager::managedMemoryAddresses[i]);
    const auto oldAddr = MemoryManager::managedMemoryAddresses[i];

    MemoryManager::managedMemoryAddresses[i] = newAddr;
    MemoryManager::managedMemoryAddressToSizeMap[newAddr] = oldManagedMemoryAddressToSizeMap[oldAddr];
    MemoryManager::managedMemoryAddressToIndexMap[newAddr] = i;

    if (MemoryManager::applicationInputs.count(oldAddr) > 0) {
      MemoryManager::applicationInputs.erase(oldAddr);
      MemoryManager::applicationInputs.insert(newAddr);
    }
    if (MemoryManager::applicationOutputs.count(oldAddr) > 0) {
      MemoryManager::applicationOutputs.erase(oldAddr);
      MemoryManager::applicationOutputs.insert(newAddr);
    }
  }
}

}  // namespace memopt
