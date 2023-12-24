#pragma once

#include <map>
#include <vector>

#include "customGraph.hpp"
#include "optimizationInput.hpp"

class Optimizer {
 public:
  static Optimizer *getInstance();
  Optimizer(Optimizer &other) = delete;
  void operator=(const Optimizer &) = delete;

  // Warning: the graph is executed once during profiling.
  CustomGraph profileAndOptimize(cudaGraph_t originalGraph);

 protected:
  Optimizer() = default;
  static Optimizer *instance;

 private:
  template <typename Strategy>
  CustomGraph optimize(OptimizationInput &optimizationInput) {
    Strategy strategyInstance;
    return strategyInstance.run(optimizationInput);
  }
};
