#include "../../profiling/memoryManager.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/types.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

namespace memopt {

OptimizationOutput NoOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  printOptimizationInput(input);

  OptimizationOutput output;
  output.optimal = true;

  // All managed data initially on device
  for (int i = 0; i < MemoryManager::managedMemoryAddresses.size(); i++) {
    output.arraysInitiallyAllocatedOnDevice.push_back(i);
  }

  // Add task groups
  std::vector<int> taskGroupStartNodes, taskGroupEndNodes;
  for (int i = 0; i < input.nodes.size(); i++) {
    taskGroupStartNodes.push_back(output.addEmptyNode());
    taskGroupEndNodes.push_back(output.addEmptyNode());
  }

  // Add edges between task groups
  for (const auto &[u, destinations] : input.edges) {
    for (auto v : destinations) {
      output.addEdge(taskGroupEndNodes[u], taskGroupStartNodes[v]);
    }
  }

  // Add nodes and edges inside task groups
  for (int i = 0; i < input.nodes.size(); i++) {
    auto &taskGroup = input.nodes[i];

    std::map<TaskId, int> taskIdToOutputNodeId;
    std::map<TaskId, bool> taskHasIncomingEdgeMap;
    for (TaskId u : taskGroup.nodes) {
      taskIdToOutputNodeId[u] = output.addTaskNode(u);
    }

    for (const auto &[u, destinations] : taskGroup.edges) {
      for (auto v : destinations) {
        output.addEdge(taskIdToOutputNodeId[u], taskIdToOutputNodeId[v]);
        taskHasIncomingEdgeMap[v] = true;
      }
    }

    for (auto u : taskGroup.nodes) {
      if (!taskHasIncomingEdgeMap[u]) {
        output.addEdge(taskGroupStartNodes[i], taskIdToOutputNodeId[u]);
      }

      if (taskGroup.edges[u].size() == 0) {
        output.addEdge(taskIdToOutputNodeId[u], taskGroupEndNodes[i]);
      }
    }
  }

  printOptimizationOutput(output, input.stageIndex);

  return output;
}

}  // namespace memopt
