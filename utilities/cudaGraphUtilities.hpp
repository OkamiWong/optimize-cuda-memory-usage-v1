#pragma once

#include <cuda.h>
#include <vector>
#include <map>

void extractGraphNodesAndEdges(
  cudaGraph_t graph,
  std::vector<CUgraphNode> &nodes,
  std::map<CUgraphNode, std::vector<CUgraphNode>> &edges
);

CUgraphNode getRootNode(cudaGraph_t graph);
