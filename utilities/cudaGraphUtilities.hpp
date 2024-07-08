#pragma once

#include <cuda.h>

#include <map>
#include <vector>

namespace memopt {

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges
);

cudaGraphNode_t getRootNode(cudaGraph_t graph);

std::vector<cudaGraphNode_t> getNodesWithZeroOutDegree(cudaGraph_t graph);

void getKernelNodeParams(cudaGraphNode_t kernelNode, CUDA_KERNEL_NODE_PARAMS &nodeParams);

bool compareKernelNodeFunctionHandle(cudaGraphNode_t kernelNode, CUfunction functionHandle);

}  // namespace memopt
