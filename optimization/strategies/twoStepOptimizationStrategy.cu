#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#include "../../profiling/memoryManager.hpp"
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

      firstStepInput.dataDependencyOverlapInBytes[i][j] =
        firstStepInput.dataDependencyOverlapInBytes[j][i] =
          std::accumulate(
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
  secondStepInput.prefetchingBandwidth = Constants::PREFETCHING_BANDWIDTH;
  secondStepInput.offloadingBandwidth = Constants::OFFLOADING_BANDWIDTH;

  secondStepInput.nodeExecutionOrder = firstStepOutput.nodeExecutionOrder;

  secondStepInput.nodeDurations.resize(optimizationInput.nodes.size());
  secondStepInput.nodeInputArrays.resize(optimizationInput.nodes.size());
  secondStepInput.nodeOutputArrays.resize(optimizationInput.nodes.size());

  for (int i = 0; i < optimizationInput.nodes.size(); i++) {
    auto &node = optimizationInput.nodes[firstStepOutput.nodeExecutionOrder[i]];

    secondStepInput.nodeDurations[i] = node.duration;

    for (auto arrayAddress : node.dataDependency.inputs) {
      secondStepInput.nodeInputArrays[i].insert(MemoryManager::managedMemoryAddressToIndexMap[arrayAddress]);
    }
    for (auto arrayAddress : node.dataDependency.outputs) {
      secondStepInput.nodeOutputArrays[i].insert(MemoryManager::managedMemoryAddressToIndexMap[arrayAddress]);
    }
  }

  secondStepInput.arraySizes.resize(MemoryManager::managedMemoryAddressCount);
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

CustomGraph TwoStepOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  printOptimizationInput(input);

  auto firstStepInput = convertToFirstStepInput(input);
  FirstStepSolver firstStepSolver(std::move(firstStepInput));
  auto firstStepOutput = firstStepSolver.solve();

  auto secondStepInput = convertToSecondStepInput(input, firstStepOutput);
  SecondStepSolver secondStepSolver;
  auto secondStepOutput = secondStepSolver.solve(std::move(secondStepInput));

  CustomGraph optimizedGraph;
  return optimizedGraph;
}
