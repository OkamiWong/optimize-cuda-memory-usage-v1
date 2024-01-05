#pragma once

#include <cupti.h>

#include <map>
#include <utility>

typedef std::pair<uint64_t, uint64_t> CudaGraphNodeLifetime;

typedef std::map<cudaGraphNode_t, CudaGraphNodeLifetime> CudaGraphExecutionTimeline;

class CudaGraphExecutionTimelineProfiler {
 public:
  static CudaGraphExecutionTimelineProfiler *getInstance();
  CudaGraphExecutionTimelineProfiler(CudaGraphExecutionTimelineProfiler &other) = delete;
  void operator=(const CudaGraphExecutionTimelineProfiler &) = delete;
  void initialize(cudaGraph_t graph);
  void finalize();
  void consumeActivityRecord(CUpti_Activity *record);
  void graphNodeClonedCallback(CUpti_GraphData *graphData);
  CudaGraphExecutionTimeline getTimeline();

 protected:
  CudaGraphExecutionTimelineProfiler() = default;
  static CudaGraphExecutionTimelineProfiler *instance;

 private:
  bool finalized = false;
  cudaGraph_t graph;
  std::map<uint64_t, CudaGraphNodeLifetime> graphNodeIdToLifetimeMap;
  std::map<uint64_t, uint64_t> originalNodeIdToClonedNodeIdMap;
  CUpti_SubscriberHandle subscriberHandle;
};
