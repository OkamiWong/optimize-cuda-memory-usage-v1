#pragma once

#include <set>
#include <tuple>
#include <vector>

class SecondStepSolver {
 public:
  struct Input {
    std::vector<float> nodeDurations;

    std::vector<size_t> arraySizes;
    std::set<int> applicationInputArrays, applicationOutputArrays;
    std::vector<std::set<int>> nodeInputArrays, nodeOutputArrays;

    float prefetchingBandwidth, offloadingBandwidth;

    float originalTotalRunningTime;
  };

  struct Output {
    // (Node Index, Array Index)
    typedef std::tuple<int, int> Prefetch;

    // (Starting Node Index, Array Index, Ending Node Index)
    typedef std::tuple<int, int, int> Offload;

    bool optimal;
    std::vector<int> indicesOfArraysInitiallyOnDevice;
    std::vector<Prefetch> prefetches;
    std::vector<Offload> offloadings;
  };

  Output solve(Input &&input);
};
