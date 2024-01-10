#pragma once

#include <set>
#include <tuple>
#include <vector>

class SecondStepSolver {
 public:
  struct Input {
    std::vector<int> nodeExecutionOrder;
    std::vector<float> nodeDurations;

    std::vector<size_t> arraySizes;
    std::set<int> applicationInputArrays, applicationOutputArrays;
    std::vector<std::set<int>> nodeInputArrays, nodeOutputArrays;

    float prefetchingBandwidth, offloadingBandwidth;
  };

  struct Output {
    // (Node Index, Array Index)
    typedef std::tuple<int, int> Prefetch;

    // (Starting Node Index, Array Index, Ending Node Index)
    typedef std::tuple<int, int, int> Offload;

    std::vector<int> indicesOfArraysInitiallyOnDevice;
    std::vector<Prefetch> prefetches;
    std::vector<Offload> offloadings;

    bool optimal;
  };

  Output solve(Input &&input);
};
