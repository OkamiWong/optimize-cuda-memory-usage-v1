#include "memoryManager.hpp"

void markMemorySpaceInitiallyOnDevice(void *devPtr) {
  MemoryManager::managedMemorySpacesInitiallyOnDevice.insert(devPtr);
}

void markMemorySpaceInitiallyOnDevice(std::initializer_list<void *> devPtrs) {
  for (auto &devPtr : devPtrs) {
    markMemorySpaceInitiallyOnDevice(devPtr);
  }
}
