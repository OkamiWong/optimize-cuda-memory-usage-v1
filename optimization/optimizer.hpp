#pragma once

#include <map>
#include <vector>

#include "customGraph.hpp"

class Optimizer {
 public:
  static Optimizer *getInstance();
  Optimizer(Optimizer &other) = delete;
  void operator=(const Optimizer &) = delete;

  typedef std::map<CUgraphNode, float> CuGraphNodeToKernelDurationMap;

  struct DataMovementPlan {
    enum class DataMovementRelativePosition {
      beforeKernel,
      afterKernel
    };

    struct DataMovementStep {
      CustomGraph::DataMovement dataMovement;
      DataMovementRelativePosition dataMovementRelativePosition;
      CUgraphNode dataMovementPosition;
    };

    cudaGraph_t originalGraph;
    std::vector<DataMovementStep> dataMovements;
  };

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
