#pragma once

#include <map>
#include <set>
#include <vector>

#include "../profiling/memoryManager.hpp"

struct OptimizationInput {
  // The domain of NodeId is [0, the total number of nodes).
  typedef int NodeId;

  struct LogicalNode {
    struct DataDependency {
      std::set<MemoryManager::ArrayInfo> inputs, outputs;
    };

    std::set<cudaGraphNode_t> nodes;
    std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
    float duration;
    DataDependency dataDependency;
  };

  NodeId nextNodeId = 0;

  std::vector<LogicalNode> nodes;
  std::map<NodeId, std::vector<NodeId>> edges;

  cudaGraph_t originalGraph;
};
