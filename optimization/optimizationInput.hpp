#pragma once

#include <map>
#include <vector>

struct OptimizationInput {
  // The domain of NodeId is [0, the total number of nodes).
  typedef int NodeId;

  struct LogicalVertex {
    std::vector<cudaGraphNode_t> nodes;
    std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  };

  NodeId nextNodeId = 0;

  std::vector<LogicalVertex> nodes;
  std::map<NodeId, std::vector<NodeId>> edges;
  std::map<NodeId, float> logicalNodeToDurationMap;

  cudaGraph_t originalGraph;
};
