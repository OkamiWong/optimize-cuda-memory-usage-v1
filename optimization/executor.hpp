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

  void executeOptimizedGraph(
    OptimizationOutput &optimizedGraph,
    ExecuteRandomTask executeRandomTask,
    float &runningTime,
    std::map<void *, void *> &managedDeviceArrayToHostArrayMap
  );

 protected:
  Executor() = default;
  static Executor *instance;
};
