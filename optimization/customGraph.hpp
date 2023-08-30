#pragma once

#include <cuda.h>

#include <map>
#include <vector>

struct CustomGraph {
  typedef uint64_t NodeId;

  inline static NodeId nextNodeId = 0;

  enum class NodeType {
    empty,
    kernel,
    dataMovement
  };

  struct DataMovement {
    enum class Direction {
      hostToDevice,
      deviceToHost
    };
    Direction direction;
    void* address;
    size_t size;
  };

  cudaGraph_t originalGraph;
  std::vector<NodeId> nodes;
  std::map<NodeId, std::vector<NodeId>> edges;
  std::map<NodeId, NodeType> nodeIdToNodeTypeMap;
  std::map<NodeId, CUgraphNode> nodeIdToCuGraphNodeMap;
  std::map<NodeId, DataMovement> nodeIdToDataMovementMap;
};
