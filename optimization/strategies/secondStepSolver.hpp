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
    std::vector<std::set<int>> kernelInputArrays, kernelOutputArrays;

    float prefetchingBandwidth, offloadingBandwidth;
  };

  struct Output {
    // (Kernel Index, Array Index)
    typedef std::tuple<int, int> Prefetch;

    // (Starting Kernel Index, Array Index, Ending Kernel Index)
    typedef std::tuple<int, int, int> Offload;

    std::vector<bool> initiallyOnDevice;
    std::vector<Prefetch> prefetches;
    std::vector<Offload> offloads;
  };

  SecondStepSolver(Input &&input);
  Output solve();

 private:
  Input input;
  Output output;
};
