#pragma once

#include <vector>
#include <map>

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<cudaGraphNode_t> &nodes,
  std::map<cudaGraphNode_t, std::vector<cudaGraphNode_t>> &edges
);

cudaGraphNode_t getRootNode(cudaGraph_t graph);
