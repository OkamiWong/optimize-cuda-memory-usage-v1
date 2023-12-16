#include <type_traits>

#include "optimizer.hpp"
#include "strategies/strategies.hpp"
#include "taskManager.hpp"
#include "../utilities/cudaGraphExecutionTimelineProfiler.hpp"
#include "../utilities/cudaUtilities.hpp"

Optimizer *Optimizer::instance = nullptr;

Optimizer *Optimizer::getInstance() {
  if (instance == nullptr) {
    instance = new Optimizer();
  }
  return instance;
}

typedef std::map<cudaGraphNode_t, cudaGraphNode_t> CudaGraphNodeDisjointSet;

CudaGraphExecutionTimeline getCudaGraphExecutionTimeline(cudaGraph_t graph){
  auto profiler = CudaGraphExecutionTimelineProfiler::getInstance();
  profiler->initialize(graph);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaDeviceSynchronize());

  profiler->finalize();

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaStreamDestroy(stream));

  return profiler->getTimeline();
}

void mergeConcurrentCudaGraphNodes(
  cudaGraph_t originalGraph,
  const CudaGraphExecutionTimeline &timeline,
  CudaGraphNodeDisjointSet &disjointSet
) {
}

void mergeCudaGraphNodesWithSameAnnotation(cudaGraph_t originalGraph, CudaGraphNodeDisjointSet &disjointSet) {
}

OptimizationInput constructOptimizationInput(cudaGraph_t originalGraph, const CudaGraphExecutionTimeline &timeline, const CudaGraphNodeDisjointSet &disjointSet) {
}

CustomGraph Optimizer::profileAndOptimize(cudaGraph_t originalGraph) {
  // Profile
  auto taskManager = TaskManager::getInstance();
  taskManager->registerDummyKernelHandle(originalGraph);

  auto timeline = getCudaGraphExecutionTimeline(originalGraph);

  CudaGraphNodeDisjointSet disjointSet;
  mergeConcurrentCudaGraphNodes(originalGraph, timeline, disjointSet);
  mergeCudaGraphNodesWithSameAnnotation(originalGraph, disjointSet);

  auto optimizationInput = constructOptimizationInput(originalGraph, timeline, disjointSet);

  // Optimize
  auto customGraph = this->optimize<TwoStepOptimizationStrategy>(optimizationInput);
  return customGraph;
}
