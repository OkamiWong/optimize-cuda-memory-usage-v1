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
    ArrayId arrayId;
  };

  bool optimal;

  std::vector<int> nodes;
  std::map<int, std::vector<int>> edges;
  std::map<int, NodeType> nodeIdToNodeTypeMap;
  std::map<int, TaskId> nodeIdToTaskIdMap;
  std::map<int, DataMovement> nodeIdToDataMovementMap;

  std::vector<ArrayId> arraysInitiallyAllocatedOnDevice;

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

  int addDataMovementNode(DataMovement::Direction direction, ArrayId arrayId, int start, int end) {
    DataMovement dataMovement;
    dataMovement.direction = direction;
    dataMovement.arrayId = arrayId;

    auto u = this->addDataMovementNode(dataMovement);
    this->addEdge(start, u);
    this->addEdge(u, end);

    return u;
  }

  void addEdge(int from, int to) {
    this->edges[from].push_back(to);
  }
};
