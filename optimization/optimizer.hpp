#pragma once

#include <map>
#include <vector>

#include "customGraph.hpp"

struct DataMovementPlan {
  enum class DataMovementRelativePosition {
    beforeKernel,
    afterKernel
  };

  cudaGraph_t originalGraph;
  std::vector<CustomGraph::DataMovement> dataMovements;
  std::vector<DataMovementRelativePosition> dataMovementRelativePositions;
  std::vector<CUgraphNode> dataMovementPositions;
};

class Optimizer {
 public:
  static Optimizer *getInstance();
  Optimizer(Optimizer &other) = delete;
  void operator=(const Optimizer &) = delete;

  typedef std::map<CUgraphNode, float> CuGraphNodeToKernelDurationMap;

  CustomGraph profileAndOptimize(cudaGraph_t originalGraph);

 protected:
  Optimizer() = default;
  static Optimizer *instance;

 private:
  template <typename Strategy>
  DataMovementPlan optimize(cudaGraph_t originalGraph, CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap) {
    Strategy strategyInstance;
    return strategyInstance.calculateDataMovementPlan(originalGraph, cuGraphNodeToKernelDurationMap);
  }

  CustomGraph transformDataMovementPlanToCustomGraph(DataMovementPlan dataMovementPlan);
};
