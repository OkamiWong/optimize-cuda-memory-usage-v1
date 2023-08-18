#pragma once

#include <cupti.h>

#include <tuple>
#include <vector>

class PeakMemoryProfiler {
 public:
  static PeakMemoryProfiler *getInstance();
  PeakMemoryProfiler(PeakMemoryProfiler &other) = delete;
  void operator=(const PeakMemoryProfiler &) = delete;
  void initialize();
  void finalize();
  void consumeActivityRecord(CUpti_Activity *record);
  uint64_t getPeakMemoryUsage();

 protected:
  PeakMemoryProfiler() = default;
  static PeakMemoryProfiler *instance;

 private:
  bool finalized = false;
  std::vector<std::tuple<uint64_t, uint64_t, int>> memoryActivityRecords;
  uint64_t currentAllocatedMemory = 0;
  uint64_t currentPeakAllocatedMemory = 0;
};
