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
  // Output:
  // - Disjoint Set: map<cudaGraphNode_t, cudaGraphNode_t> nodeToDisjointSetRootMap

  // Merge nodes that run concurrently
  // Design:
  // - Iterate through all nodes, merge each node with their root in the Disjoint Set
  // Output:
  // - 

  // Find the duration of nodes 
  // Design:
  // - Reuse the node starting times and end times obtained in the first step
  // - The duration of a logical node is the maximum of end times of corresponding actual nodes minus the minimum of starting times

  // Optimize
  // Design:
  // - Integer Programming solved via Z3 / OR-Tools (SCIP, GLPK, ...)

  return customGraph;
}