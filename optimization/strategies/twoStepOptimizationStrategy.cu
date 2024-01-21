#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#include "../../profiling/memoryManager.hpp"
#include "../../utilities/configurationManager.hpp"
#include "../../utilities/constants.hpp"
#include "../../utilities/logger.hpp"
#include "firstStepSolver.hpp"
#include "secondStepSolver.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

FirstStepSolver::Input convertToFirstStepInput(OptimizationInput &optimizationInput) {
  const auto numLogicalNodes = optimizationInput.nodes.size();

  FirstStepSolver::Input firstStepInput;

  firstStepInput.n = numLogicalNodes;
  firstStepInput.edges.resize(numLogicalNodes);
  firstStepInput.dataDependencyOverlapInBytes.resize(numLogicalNodes, std::vector<size_t>(numLogicalNodes, 0));

  // Copy edges
  for (int i = 0; i < numLogicalNodes; i++) {
    firstStepInput.edges[i] = optimizationInput.edges[i];
  }

  // Calculate data dependency overlaps
  for (int i = 0; i < numLogicalNodes; i++) {
    const auto &u = optimizationInput.nodes[i];
    std::set<void *> uTotalDataDependency;
    std::set_union(
      u.dataDependency.inputs.begin(),
      u.dataDependency.inputs.end(),
      u.dataDependency.outputs.begin(),
      u.dataDependency.outputs.end(),
      std::inserter(uTotalDataDependency, uTotalDataDependency.begin())
    );

    for (int j = i + 1; j < numLogicalNodes; j++) {
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
  secondStepInput.prefetchingBandwidth = ConfigurationManager::getConfiguration().prefetchingBandwidthInGB * 1e9;
  secondStepInput.offloadingBandwidth = ConfigurationManager::getConfiguration().prefetchingBandwidthInGB * 1e9;
  secondStepInput.originalTotalRunningTime = optimizationInput.originalTotalRunningTime;

  secondStepInput.nodeDurations.resize(optimizationInput.nodes.size());
  secondStepInput.nodeInputArrays.resize(optimizationInput.nodes.size());
  secondStepInput.nodeOutputArrays.resize(optimizationInput.nodes.size());

  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    // Second step assumes logical nodes are sorted in execution order.
    auto &node = optimizationInput.nodes[firstStepOutput.nodeExecutionOrder[i]];

    secondStepInput.nodeDurations[i] = node.duration;

    for (auto arrayAddress : node.dataDependency.inputs) {
      secondStepInput.nodeInputArrays[i].insert(MemoryManager::managedMemoryAddressToIndexMap[arrayAddress]);
    }
    for (auto arrayAddress : node.dataDependency.outputs) {
      secondStepInput.nodeOutputArrays[i].insert(MemoryManager::managedMemoryAddressToIndexMap[arrayAddress]);
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

CustomGraph convertToCustomGraph(
  OptimizationInput &optimizationInput,
  FirstStepSolver::Output &firstStepOutput,
  SecondStepSolver::Output &secondStepOutput
) {
  CustomGraph optimizedGraph;

  if (!secondStepOutput.optimal) {
    optimizedGraph.optimal = false;
    return optimizedGraph;
  }

  optimizedGraph.optimal = true;
  optimizedGraph.originalGraph = optimizationInput.originalGraph;

  // Add arrays that should be on device initially
  for (auto index : secondStepOutput.indicesOfArraysInitiallyOnDevice) {
    auto addr = MemoryManager::managedMemoryAddresses[index];
    optimizedGraph.arraysInitiallyAllocatedOnDevice.push_back(std::make_pair(
      addr,
      MemoryManager::managedMemoryAddressToSizeMap[addr]
    ));
  }

  // Add logical nodes
  std::vector<CustomGraph::NodeId> logicalNodeStarts, logicalNodeBodies, logicalNodeEnds;
  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    logicalNodeStarts.push_back(optimizedGraph.addEmptyNode());
    logicalNodeBodies.push_back(optimizedGraph.addEmptyNode());
    logicalNodeEnds.push_back(optimizedGraph.addEmptyNode());
  }

  // Add edges between logical ndoes
  for (int i = 1; i < firstStepOutput.nodeExecutionOrder.size(); i++) {
    const auto previousNodeIndex = firstStepOutput.nodeExecutionOrder[i - 1];
    const auto currentNodeIndex = firstStepOutput.nodeExecutionOrder[i];
    optimizedGraph.addEdge(
      logicalNodeEnds[previousNodeIndex],
      logicalNodeStarts[currentNodeIndex]
    );
  }

  // Add nodes and edges inside logical nodes
  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    auto &logicalNode = optimizationInput.nodes[i];

    std::map<cudaGraphNode_t, CustomGraph::NodeId> cudaGraphNodeToCustomGraphNodeIdMap;
    std::map<cudaGraphNode_t, bool> cudaGraphNodeHasIncomingEdgeMap;
    for (auto u : logicalNode.nodes) {
      cudaGraphNodeToCustomGraphNodeIdMap[u] = optimizedGraph.addKernelNode(u);
    }

    for (const auto &[u, destinations] : logicalNode.edges) {
      for (auto v : destinations) {
        optimizedGraph.addEdge(cudaGraphNodeToCustomGraphNodeIdMap[u], cudaGraphNodeToCustomGraphNodeIdMap[v]);
        cudaGraphNodeHasIncomingEdgeMap[v] = true;
      }
    }

    optimizedGraph.addEdge(logicalNodeStarts[i], logicalNodeBodies[i]);

    for (auto u : logicalNode.nodes) {
      if (!cudaGraphNodeHasIncomingEdgeMap[u]) {
        optimizedGraph.addEdge(logicalNodeBodies[i], cudaGraphNodeToCustomGraphNodeIdMap[u]);
      }

      if (logicalNode.edges[u].size() == 0) {
        optimizedGraph.addEdge(cudaGraphNodeToCustomGraphNodeIdMap[u], logicalNodeEnds[i]);
      }
    }
  }

  // Add prefetches
  for (const auto &[startingNodeIndex, arrayIndex] : secondStepOutput.prefetches) {
    void *arrayAddress = MemoryManager::managedMemoryAddresses[arrayIndex];
    size_t arraySize = MemoryManager::managedMemoryAddressToSizeMap[arrayAddress];

    int endingNodeIndex = startingNodeIndex;
    while (endingNodeIndex < firstStepOutput.nodeExecutionOrder.size()) {
      auto &logicalNode = optimizationInput.nodes[firstStepOutput.nodeExecutionOrder[endingNodeIndex]];
      if (logicalNode.dataDependency.inputs.count(arrayAddress) > 0 || logicalNode.dataDependency.outputs.count(arrayAddress) > 0) {
        break;
      }
      endingNodeIndex++;
    }

    // Ignore unnecessary prefetch, which has no dependent kernels after it.
    if (endingNodeIndex == firstStepOutput.nodeExecutionOrder.size()) {
      continue;
    }

    optimizedGraph.addDataMovementNode(
      CustomGraph::DataMovement::Direction::hostToDevice,
      arrayAddress,
      arraySize,
      logicalNodeStarts[firstStepOutput.nodeExecutionOrder[startingNodeIndex]],
      logicalNodeBodies[firstStepOutput.nodeExecutionOrder[endingNodeIndex]]
    );
  }

  // Add offloadings
  for (const auto &[startingNodeIndex, arrayIndex, endingNodeIndex] : secondStepOutput.offloadings) {
    void *arrayAddress = MemoryManager::managedMemoryAddresses[arrayIndex];
    size_t arraySize = MemoryManager::managedMemoryAddressToSizeMap[arrayAddress];

    optimizedGraph.addDataMovementNode(
      CustomGraph::DataMovement::Direction::deviceToHost,
      arrayAddress,
      arraySize,
      logicalNodeEnds[firstStepOutput.nodeExecutionOrder[startingNodeIndex]],
      logicalNodeStarts[firstStepOutput.nodeExecutionOrder[endingNodeIndex]]
    );
  }

  return optimizedGraph;
}

CustomGraph TwoStepOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  printOptimizationInput(input);

  auto firstStepInput = convertToFirstStepInput(input);
  FirstStepSolver firstStepSolver(std::move(firstStepInput));
  auto firstStepOutput = firstStepSolver.solve();

  auto secondStepInput = convertToSecondStepInput(input, firstStepOutput);
  SecondStepSolver secondStepSolver;
  auto secondStepOutput = secondStepSolver.solve(std::move(secondStepInput));

  return convertToCustomGraph(input, firstStepOutput, secondStepOutput);
}
