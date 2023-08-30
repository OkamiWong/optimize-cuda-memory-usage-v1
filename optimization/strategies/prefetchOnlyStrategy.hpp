#pragma once

#include "../optimizer.hpp"

class PrefetchOnlyStrategy {
 public:
  DataMovementPlan calculateDataMovementPlan(
    cudaGraph_t originalGraph,
    Optimizer::CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap
  );
};
