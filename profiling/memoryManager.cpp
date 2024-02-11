#include "memoryManager.hpp"

#include <cassert>

void updateManagedMemoryAddress(const std::map<void *, void *> oldAddressToNewAddressMap) {
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