#include <type_traits>

#include "optimizer.hpp"
#include "strategies/strategies.hpp"
#include "taskManager.hpp"

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

CustomGraph Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  // Profile
  auto taskManager = TaskManager::getInstance();
  taskManager->registerDummyKernelHandle(originalGraph);

  // Find out nodes that run concurrently
  // Design:
  // - Use CUPTI Activity API to obtain node starting times and end times
  // - Merge nodes that overlap in timeline
  // - Use Disjoint Set to maintain the set relations
  // Input:
  // - cudaGraph_t
  // - Disjoint Set
  // Output:
  // - Disjoint Set

  // Find out nodes that corresponds to the same kernel IO annotation
  // Design:
  // - DFS
  // Input:
  // - cudaGraph_t
  // - Disjoint Set
  // Output:
  // - Disjoint Set

  // Merge nodes that run concurrently
  // Design:
  // - Iterate through all nodes, merge each node with their root in the Disjoint Set
  // - Dependencies between merged nodes are stored.
  // Input:
  // - cudaGraph_t
  // - Disjoint Set
  // Output:
  // - OptimizationInput

  // Find the duration of nodes 
  // Design:
  // - Reuse the node starting times and end times obtained in the first step
  // - The duration of a logical node is the maximum of end times of corresponding actual nodes minus the minimum of starting times
  // Input:
  // - OptimizationInput
  // Output:
  // - OptimizationInput

  // Optimize
  // Design:
  // - Integer Programming solved via Z3 / OR-Tools (SCIP, GLPK, ...)
  // Input:
  // - OptimizationInput
  // Output:
  // - CustonGraph

  return customGraph;
}