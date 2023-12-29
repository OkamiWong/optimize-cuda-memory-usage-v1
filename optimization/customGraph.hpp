#pragma once

#include <cuda.h>

#include <map>
#include <vector>

struct CustomGraph {
  // The domain of NodeId is [0, the total number of nodes).
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
  std::map<NodeId, cudaGraphNode_t> nodeIdToCuGraphNodeMap;
  std::map<NodeId, DataMovement> nodeIdToDataMovementMap;

  NodeId addEmptyNode() {
    auto u = nextNodeId++;
    this->nodes.push_back(u);
    this->nodeIdToNodeTypeMap[u] = NodeType::empty;
    return u;
  }

  NodeId addKernelNode(cudaGraphNode_t nodeInOriginalGraph) {
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

  NodeId addDataMovementNode(DataMovement::Direction direction, void* address, size_t size, NodeId start, NodeId end) {
    DataMovement dataMovement;
    dataMovement.direction = direction;
    dataMovement.address = address;
    dataMovement.size = size;

    auto u = this->addDataMovementNode(dataMovement);
    this->addEdge(start, u);
    this->addEdge(u, end);

    return u;
  }

  void addEdge(NodeId from, NodeId to) {
    this->edges[from].push_back(to);
  }
};
