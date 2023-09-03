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

  CustomGraph profileAndOptimize(cudaGraph_t originalGraph);

 protected:
  Optimizer() = default;
  static Optimizer *instance;

 private:
  template <typename Strategy>
  CustomGraph optimize(cudaGraph_t originalGraph, CuGraphNodeToKernelDurationMap cuGraphNodeToKernelDurationMap) {
    Strategy strategyInstance;
    return strategyInstance.run(originalGraph, cuGraphNodeToKernelDurationMap);
  }
};
