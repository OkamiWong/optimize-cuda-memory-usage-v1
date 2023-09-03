#pragma once

#include <cuda.h>

#include <map>
#include <vector>

#include "../../profiling/memoryManager.hpp"

struct KernelDataDependency {
  std::vector<MemoryManager::ArrayInfo> inputs, outputs;
};

std::map<CUgraphNode, KernelDataDependency> mapKernelOntoDataDependency(
  std::vector<CUgraphNode>& nodes,
  std::map<CUgraphNode, std::vector<CUgraphNode>>& edges
);
