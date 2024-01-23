#pragma once

#include <functional>
#include <map>

#include "optimizationOutput.hpp"

typedef std::function<void(int, std::map<void *, void *>, cudaStream_t)> ExecuteRandomTask;

class Executor {
 public:
  static Executor *getInstance();
  Executor(Executor &other) = delete;
  void operator=(const Executor &) = delete;

  // New CUDA Graph running time in seconds is returned
  float executeOptimizedGraph(OptimizationOutput &optimizedGraph, ExecuteRandomTask executeRandomTask);

 protected:
  Executor() = default;
  static Executor *instance;
};
