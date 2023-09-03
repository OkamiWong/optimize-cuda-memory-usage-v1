#pragma once

#include <cuda.h>

#include <map>
#include <vector>

struct CustomGraph {
  typedef int NodeId;

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

  NodeId nextNodeId = 0;

  cudaGraph_t originalGraph;
  std::vector<NodeId> nodes;
  std::map<NodeId, std::vector<NodeId>> edges;
  std::map<NodeId, NodeType> nodeIdToNodeTypeMap;
  std::map<NodeId, CUgraphNode> nodeIdToCuGraphNodeMap;
  std::map<NodeId, DataMovement> nodeIdToDataMovementMap;

  NodeId addEmptyNode() {
    auto u = nextNodeId++;
    this->nodes.push_back(u);
    this->nodeIdToNodeTypeMap[u] = NodeType::empty;
    return u;
  }

  NodeId addKernelNode(CUgraphNode nodeInOriginalGraph) {
    auto u = this->addEmptyNode();
    this->nodeIdToNodeTypeMap[u] = NodeType::kernel;
    this->nodeIdToCuGraphNodeMap[u] = nodeInOriginalGraph;
    return u;
  }

  NodeId addDataMovementNode(DataMovement dataMovement) {
    auto u = this->addEmptyNode();
    this->nodeIdToNodeTypeMap[u] = NodeType::dataMovement;
    this->nodeIdToDataMovementMap[u] = dataMovement;
    return u;
  }

  void addEdge(NodeId from, NodeId to) {
    this->edges[from].push_back(to);
  }
};
