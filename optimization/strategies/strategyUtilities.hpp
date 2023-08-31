#pragma once

#include <cuda.h>

#include <map>
#include <vector>

#include "../../profiling/memoryManager.hpp"

struct KernelDataDependency {
  std::vector<MemoryManager::ArrayInfo> inputs, outputs;
};

std::map<CUgraphNode, KernelDataDependency> mapKernelOntoDataDependency(
  const std::vector<CUgraphNode>& nodes,
  const std::map<CUgraphNode, std::vector<CUgraphNode>>& edges
);
