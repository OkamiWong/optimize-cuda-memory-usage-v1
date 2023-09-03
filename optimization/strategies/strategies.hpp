#pragma once

#include "../optimizer.hpp"

class PrefetchOnlyStrategy {
 public:
  CustomGraph run(
    cudaGraph_t originalGraph,
    Optimizer::CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap
  );
};
