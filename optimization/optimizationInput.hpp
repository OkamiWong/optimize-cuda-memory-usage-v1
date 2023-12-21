#pragma once

#include <map>
#include <vector>

#include "../profiling/memoryManager.hpp"

struct NodeDataDependency {
  std::vector<MemoryManager::ArrayInfo> inputs, outputs;
};

struct OptimizationInput {
  // The domain of NodeId is [0, the total number of nodes).
  typedef int NodeId;

  struct LogicalVertex {
    std::vector<cudaGraphNode_t> nodes;
    std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
    float duration;
    NodeDataDependency dataDependency;
  };

  NodeId nextNodeId = 0;

  std::vector<LogicalVertex> nodes;
  std::map<NodeId, std::vector<NodeId>> edges;

  cudaGraph_t originalGraph;
};
