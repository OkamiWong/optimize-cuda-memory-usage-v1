#pragma once

#include <map>
#include <utility>
#include <vector>

#include "../utilities/types.hpp"

struct OptimizationOutput {
  enum class NodeType {
    empty,
    task,
    dataMovement
  };

  struct DataMovement {
    enum class Direction {
      hostToDevice,
      deviceToHost
    };
    Direction direction;
    void* address;
  };

  bool optimal;

  std::vector<int> nodes;
  std::map<int, std::vector<int>> edges;
  std::map<int, NodeType> nodeIdToNodeTypeMap;
  std::map<int, TaskId> nodeIdToTaskIdMap;
  std::map<int, DataMovement> nodeIdToDataMovementMap;

  std::vector<void*> arraysInitiallyAllocatedOnDevice;

  int addEmptyNode() {
    auto u = this->nodes.size();
    this->nodes.push_back(u);
    this->nodeIdToNodeTypeMap[u] = NodeType::empty;
    return u;
  }

  int addTaskNode(TaskId taskId) {
    auto u = this->addEmptyNode();
    this->nodeIdToNodeTypeMap[u] = NodeType::task;
    this->nodeIdToTaskIdMap[u] = taskId;
    return u;
  }

  int addDataMovementNode(DataMovement dataMovement) {
    auto u = this->addEmptyNode();
    this->nodeIdToNodeTypeMap[u] = NodeType::dataMovement;
    this->nodeIdToDataMovementMap[u] = dataMovement;
    return u;
  }

  int addDataMovementNode(DataMovement::Direction direction, void* address, int start, int end) {
    DataMovement dataMovement;
    dataMovement.direction = direction;
    dataMovement.address = address;

    auto u = this->addDataMovementNode(dataMovement);
    this->addEdge(start, u);
    this->addEdge(u, end);

    return u;
  }

  void addEdge(int from, int to) {
    this->edges[from].push_back(to);
  }
};
