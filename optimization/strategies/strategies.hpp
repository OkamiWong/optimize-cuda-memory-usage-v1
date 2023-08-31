#pragma once

#include "../optimizer.hpp"

class PrefetchOnlyStrategy {
 public:
  Optimizer::DataMovementPlan calculateDataMovementPlan(
    cudaGraph_t originalGraph,
    Optimizer::CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap
  );
};
