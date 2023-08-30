#pragma once

#include "customGraph.hpp"

class Optimizer {
 public:
  static Optimizer *getInstance();
  Optimizer(Optimizer &other) = delete;
  void operator=(const Optimizer &) = delete;

  CustomGraph profileAndOptimize(cudaGraph_t originalGraph);

 protected:
  Optimizer() = default;
  static Optimizer *instance;
};
