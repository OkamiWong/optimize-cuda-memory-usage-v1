#pragma once

#include <set>
#include <tuple>
#include <vector>

#include "../../utilities/types.hpp"

class SecondStepSolver {
 public:
  struct Input {
    std::vector<float> taskGroupRunningTimes;

    std::vector<size_t> arraySizes;
    std::set<ArrayId> applicationInputArrays, applicationOutputArrays;
    std::vector<std::set<ArrayId>> taskGroupInputArrays, taskGroupOutputArrays;

    float prefetchingBandwidth, offloadingBandwidth;

    bool optimizeForRepetitiveExecution;

    float originalTotalRunningTime;
  };

  struct Output {
    // (Task Group Index, Array Index)
    // Prefetch at the start of a task group
    typedef std::tuple<TaskGroupId, ArrayId> Prefetch;

    // (Starting Task Group Index, Array Index, Ending Task Group Index)
    // Offload at the end of a task group
    typedef std::tuple<TaskGroupId, ArrayId, TaskGroupId> Offload;

    bool optimal;
    std::vector<ArrayId> indicesOfArraysInitiallyOnDevice;
    std::vector<Prefetch> prefetches;
    std::vector<Offload> offloadings;
  };

  Output solve(Input &&input);
};
