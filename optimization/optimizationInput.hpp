#pragma once

#include <map>
#include <set>
#include <vector>

#include "../utilities/types.hpp"

namespace memopt {

struct OptimizationInput {
  struct TaskGroup {
    struct DataDependency {
      std::set<void *> inputs, outputs;
    };

    std::set<TaskId> nodes;
    std::map<TaskId, std::vector<TaskId>> edges;
    float runningTime;
    DataDependency dataDependency;
  };

  std::vector<TaskGroup> nodes;
  std::map<TaskGroupId, std::vector<TaskGroupId>> edges;

  float originalTotalRunningTime;

  bool specifyArraysInitiallyAllocatedOnDevice;
  std::vector<ArrayId> arraysInitiallyAllocatedOnDevice;

  int stageIndex;
};

}  // namespace memopt
