#pragma once

#include <map>
#include <set>
#include <vector>

// Technical debt: The array size can be removed from ArrayInfo,
// because it can be inferred from MemoryManager.
typedef std::tuple<void *, size_t> ArrayInfo;

struct OptimizationInput {
  // The domain of NodeId is [0, the total number of nodes).
  typedef int NodeId;

  struct LogicalNode {
    struct DataDependency {
      std::set<ArrayInfo> inputs, outputs;
    };

    std::set<cudaGraphNode_t> nodes;
    std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
    float duration;
    DataDependency dataDependency;
  };

  std::vector<LogicalNode> nodes;
  std::map<NodeId, std::vector<NodeId>> edges;

  cudaGraph_t originalGraph;
};
