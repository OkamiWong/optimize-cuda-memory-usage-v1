#include <cuda.h>

#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "cudaGraphUtilities.hpp"
#include "cudaUtilities.hpp"

namespace memopt {

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges
) {
  size_t numNodes, numEdges;
  checkCudaErrors(cudaGraphGetNodes(graph, nullptr, &numNodes));
  checkCudaErrors(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  auto rawNodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
  auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
  auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
  checkCudaErrors(cudaGraphGetNodes(graph, rawNodes.get(), &numNodes));
  checkCudaErrors(cudaGraphGetEdges(graph, from.get(), to.get(), &numEdges));

  nodes.clear();
  for (int i = 0; i < numNodes; i++) {
    nodes.push_back(rawNodes[i]);
  }

  edges.clear();
  for (int i = 0; i < numEdges; i++) {
    edges[from[i]].push_back(to[i]);
  }
}

cudaGraphNode_t getRootNode(cudaGraph_t graph) {
  size_t numRootNodes;
  checkCudaErrors(cudaGraphGetRootNodes(graph, nullptr, &numRootNodes));
  assert(numRootNodes == 1);

  auto rootNodes = std::make_unique<cudaGraphNode_t[]>(numRootNodes);
  checkCudaErrors(cudaGraphGetRootNodes(graph, rootNodes.get(), &numRootNodes));
  return rootNodes[0];
}

std::vector<cudaGraphNode_t> getNodesWithZeroOutDegree(cudaGraph_t graph) {
  std::vector<cudaGraphNode_t> nodesWithZeroOutDegree;

  std::vector<cudaGraphNode_t> nodes;
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> edges;
  extractGraphNodesAndEdges(graph, nodes, edges);

  for (auto u : nodes) {
    if (edges[u].size() == 0) {
      nodesWithZeroOutDegree.push_back(u);
    }
  }

  return nodesWithZeroOutDegree;
}

void getKernelNodeParams(cudaGraphNode_t kernelNode, CUDA_KERNEL_NODE_PARAMS &nodeParams) {
  cudaGraphNodeType nodeType;
  checkCudaErrors(cudaGraphNodeGetType(kernelNode, &nodeType));
  assert(nodeType == cudaGraphNodeTypeKernel);

  // Why switch to driver API:
  // https://forums.developer.nvidia.com/t/cuda-runtime-api-error-for-cuda-graph-and-opencv/215408/13
  checkCudaErrors(cuGraphKernelNodeGetParams(kernelNode, &nodeParams));
}

bool compareKernelNodeFunctionHandle(cudaGraphNode_t kernelNode, CUfunction functionHandle) {
  cudaGraphNodeType nodeType;
  checkCudaErrors(cudaGraphNodeGetType(kernelNode, &nodeType));
  if (nodeType == cudaGraphNodeTypeKernel) {
    CUDA_KERNEL_NODE_PARAMS nodeParams;
    getKernelNodeParams(kernelNode, nodeParams);

    if (nodeParams.func == functionHandle) {
      return true;
    }
  }
  return false;
}

}  // namespace memopt
