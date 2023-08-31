#include <cuda.h>

#include "strategyUtilities.hpp"

std::map<CUgraphNode, KernelDataDependency> mapKernelOntoDataDependency(
  const std::vector<CUgraphNode>& nodes,
  const std::map<CUgraphNode, std::vector<CUgraphNode>>& edges
) {
  // TODO
}
