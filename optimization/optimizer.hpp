#pragma once

#include <map>
#include <vector>

#include "optimizationInput.hpp"
#include "optimizationOutput.hpp"

namespace memopt {

class Optimizer {
 public:
  static Optimizer *getInstance();
  Optimizer(Optimizer &other) = delete;
  void operator=(const Optimizer &) = delete;

  // Warning: the graph is executed once during profiling.
  OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph);

 protected:
  Optimizer() = default;
  static Optimizer *instance;

 private:
  template <typename Strategy>
  OptimizationOutput optimize(OptimizationInput &optimizationInput) {
    Strategy strategyInstance;
    return strategyInstance.run(optimizationInput);
  }
};

}  // namespace memopt
