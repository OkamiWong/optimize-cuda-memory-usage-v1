#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#include "../../profiling/memoryManager.hpp"
#include "../../utilities/configurationManager.hpp"
#include "../../utilities/constants.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/types.hpp"
#include "firstStepSolver.hpp"
#include "secondStepSolver.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

FirstStepSolver::Input convertToFirstStepInput(OptimizationInput &optimizationInput) {
  const auto numTaskGroups = optimizationInput.nodes.size();

  FirstStepSolver::Input firstStepInput;

  firstStepInput.n = numTaskGroups;
  firstStepInput.edges.resize(numTaskGroups);
  firstStepInput.dataDependencyOverlapInBytes.resize(numTaskGroups, std::vector<size_t>(numTaskGroups, 0));

  // Copy edges
  for (int i = 0; i < numTaskGroups; i++) {
    firstStepInput.edges[i] = optimizationInput.edges[i];
  }

  // Calculate data dependency overlaps
  for (int i = 0; i < numTaskGroups; i++) {
    const auto &u = optimizationInput.nodes[i];
    std::set<void *> uTotalDataDependency;
    std::set_union(
      u.dataDependency.inputs.begin(),
      u.dataDependency.inputs.end(),
      u.dataDependency.outputs.begin(),
      u.dataDependency.outputs.end(),
      std::inserter(uTotalDataDependency, uTotalDataDependency.begin())
    );

    for (int j = i + 1; j < numTaskGroups; j++) {
      const auto &v = optimizationInput.nodes[j];

      std::set<void *> vTotalDataDependency;
      std::set_union(
        v.dataDependency.inputs.begin(),
        v.dataDependency.inputs.end(),
        v.dataDependency.outputs.begin(),
        v.dataDependency.outputs.end(),
        std::inserter(vTotalDataDependency, vTotalDataDependency.begin())
      );

      std::set<void *> dataDependencyIntersection;
      std::set_intersection(
        uTotalDataDependency.begin(),
        uTotalDataDependency.end(),
        vTotalDataDependency.begin(),
        vTotalDataDependency.end(),
        std::inserter(dataDependencyIntersection, dataDependencyIntersection.begin())
      );

      firstStepInput.dataDependencyOverlapInBytes[i][j] = firstStepInput.dataDependencyOverlapInBytes[j][i] = std::accumulate(
        dataDependencyIntersection.begin(),
        dataDependencyIntersection.end(),
        static_cast<size_t>(0),
        [](size_t a, void *b) {
          return a + MemoryManager::managedMemoryAddressToSizeMap[b];
        }
      );
    }
  }

  return firstStepInput;
}

SecondStepSolver::Input convertToSecondStepInput(OptimizationInput &optimizationInput, FirstStepSolver::Output &firstStepOutput) {
  SecondStepSolver::Input secondStepInput;
  secondStepInput.prefetchingBandwidth = ConfigurationManager::getConfig().prefetchingBandwidthInGB * 1e9;
  secondStepInput.offloadingBandwidth = ConfigurationManager::getConfig().prefetchingBandwidthInGB * 1e9;
  secondStepInput.originalTotalRunningTime = optimizationInput.originalTotalRunningTime;

  secondStepInput.taskGroupRunningTimes.resize(optimizationInput.nodes.size());
  secondStepInput.taskGroupInputArrays.resize(optimizationInput.nodes.size());
  secondStepInput.taskGroupOutputArrays.resize(optimizationInput.nodes.size());

  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    // Second step assumes task groups are sorted in execution order.
    auto &node = optimizationInput.nodes[firstStepOutput.taskGroupExecutionOrder[i]];

    secondStepInput.taskGroupRunningTimes[i] = node.runningTime;

    for (auto arrayAddress : node.dataDependency.inputs) {
      secondStepInput.taskGroupInputArrays[i].insert(MemoryManager::managedMemoryAddressToIndexMap[arrayAddress]);
    }
    for (auto arrayAddress : node.dataDependency.outputs) {
      secondStepInput.taskGroupOutputArrays[i].insert(MemoryManager::managedMemoryAddressToIndexMap[arrayAddress]);
    }
  }

  secondStepInput.arraySizes.resize(MemoryManager::managedMemoryAddresses.size());
  for (const auto &[ptr, index] : MemoryManager::managedMemoryAddressToIndexMap) {
    secondStepInput.arraySizes[index] = MemoryManager::managedMemoryAddressToSizeMap[ptr];
  }

  for (auto ptr : MemoryManager::applicationInputs) {
    secondStepInput.applicationInputArrays.insert(MemoryManager::managedMemoryAddressToIndexMap[ptr]);
  }
  for (auto ptr : MemoryManager::applicationOutputs) {
    secondStepInput.applicationOutputArrays.insert(MemoryManager::managedMemoryAddressToIndexMap[ptr]);
  }

  return secondStepInput;
}

OptimizationOutput convertToCustomGraph(
  OptimizationInput &optimizationInput,
  FirstStepSolver::Output &firstStepOutput,
  SecondStepSolver::Output &secondStepOutput
) {
  OptimizationOutput optimizedGraph;

  if (!secondStepOutput.optimal) {
    optimizedGraph.optimal = false;
    return optimizedGraph;
  }

  optimizedGraph.optimal = true;

  // Add arrays that should be on device initially
  for (auto index : secondStepOutput.indicesOfArraysInitiallyOnDevice) {
    auto addr = MemoryManager::managedMemoryAddresses[index];
    optimizedGraph.arraysInitiallyAllocatedOnDevice.push_back(std::make_pair(
      addr,
      MemoryManager::managedMemoryAddressToSizeMap[addr]
    ));
  }

  // Add task groups
  std::vector<int> taskGroupStartNodes, taskGroupBodyNodes, taskGroupEndNodes;
  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    taskGroupStartNodes.push_back(optimizedGraph.addEmptyNode());
    taskGroupBodyNodes.push_back(optimizedGraph.addEmptyNode());
    taskGroupEndNodes.push_back(optimizedGraph.addEmptyNode());
  }

  // Add edges between task groups
  for (int i = 1; i < firstStepOutput.taskGroupExecutionOrder.size(); i++) {
    const auto previousTaskGroupId = firstStepOutput.taskGroupExecutionOrder[i - 1];
    const auto currentTaskGroupId = firstStepOutput.taskGroupExecutionOrder[i];
    optimizedGraph.addEdge(
      taskGroupEndNodes[previousTaskGroupId],
      taskGroupStartNodes[currentTaskGroupId]
    );
  }

  // Add nodes and edges inside task groups
  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    auto &taskGroup = optimizationInput.nodes[i];

    std::map<TaskId, int> taskIdToOutputNodeIdMap;
    std::map<TaskId, bool> taskHasIncomingEdgeMap;
    for (auto u : taskGroup.nodes) {
      taskIdToOutputNodeIdMap[u] = optimizedGraph.addTaskNode(u);
    }

    for (const auto &[u, destinations] : taskGroup.edges) {
      for (auto v : destinations) {
        optimizedGraph.addEdge(taskIdToOutputNodeIdMap[u], taskIdToOutputNodeIdMap[v]);
        taskHasIncomingEdgeMap[v] = true;
      }
    }

    optimizedGraph.addEdge(taskGroupStartNodes[i], taskGroupBodyNodes[i]);

    for (auto u : taskGroup.nodes) {
      if (!taskHasIncomingEdgeMap[u]) {
        optimizedGraph.addEdge(taskGroupBodyNodes[i], taskIdToOutputNodeIdMap[u]);
      }

      if (taskGroup.edges[u].size() == 0) {
        optimizedGraph.addEdge(taskIdToOutputNodeIdMap[u], taskGroupEndNodes[i]);
      }
    }
  }

  // Add prefetches
  for (const auto &[startingNodeIndex, arrayIndex] : secondStepOutput.prefetches) {
    void *arrayAddress = MemoryManager::managedMemoryAddresses[arrayIndex];
    size_t arraySize = MemoryManager::managedMemoryAddressToSizeMap[arrayAddress];

    int endingNodeIndex = startingNodeIndex;
    while (endingNodeIndex < firstStepOutput.taskGroupExecutionOrder.size()) {
      auto &taskGroup = optimizationInput.nodes[firstStepOutput.taskGroupExecutionOrder[endingNodeIndex]];
      if (taskGroup.dataDependency.inputs.count(arrayAddress) > 0 || taskGroup.dataDependency.outputs.count(arrayAddress) > 0) {
        break;
      }
      endingNodeIndex++;
    }

    // Ignore unnecessary prefetch, which has no dependent kernels after it.
    if (endingNodeIndex == firstStepOutput.taskGroupExecutionOrder.size()) {
      continue;
    }

    optimizedGraph.addDataMovementNode(
      OptimizationOutput::DataMovement::Direction::hostToDevice,
      arrayAddress,
      taskGroupStartNodes[firstStepOutput.taskGroupExecutionOrder[startingNodeIndex]],
      taskGroupBodyNodes[firstStepOutput.taskGroupExecutionOrder[endingNodeIndex]]
    );
  }

  // Add offloadings
  for (const auto &[startingNodeIndex, arrayIndex, endingNodeIndex] : secondStepOutput.offloadings) {
    void *arrayAddress = MemoryManager::managedMemoryAddresses[arrayIndex];
    size_t arraySize = MemoryManager::managedMemoryAddressToSizeMap[arrayAddress];

    optimizedGraph.addDataMovementNode(
      OptimizationOutput::DataMovement::Direction::deviceToHost,
      arrayAddress,
      taskGroupEndNodes[firstStepOutput.taskGroupExecutionOrder[startingNodeIndex]],
      taskGroupStartNodes[firstStepOutput.taskGroupExecutionOrder[endingNodeIndex]]
    );
  }

  return optimizedGraph;
}

OptimizationOutput TwoStepOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  printOptimizationInput(input);

  auto firstStepInput = convertToFirstStepInput(input);
  FirstStepSolver firstStepSolver(std::move(firstStepInput));
  auto firstStepOutput = firstStepSolver.solve();

  auto secondStepInput = convertToSecondStepInput(input, firstStepOutput);
  SecondStepSolver secondStepSolver;
  auto secondStepOutput = secondStepSolver.solve(std::move(secondStepInput));

  auto output = convertToCustomGraph(input, firstStepOutput, secondStepOutput);

  printOptimizationOutput(output);

  return output;
}
