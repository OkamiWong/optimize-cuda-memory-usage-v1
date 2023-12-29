#include "../../utilities/logger.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

CustomGraph NoOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  CustomGraph optimizedGraph;
  optimizedGraph.originalGraph = input.originalGraph;

  // Add logical nodes
  std::vector<CustomGraph::NodeId> logicalNodeStarts, logicalNodeEnds;
  for (int i = 0; i < input.nodes.size(); i++) {
    logicalNodeStarts.push_back(optimizedGraph.addEmptyNode());
    logicalNodeEnds.push_back(optimizedGraph.addEmptyNode());
  }

  // Add edges between logical ndoes
  for (const auto &[u, destinations] : input.edges) {
    for (auto v : destinations) {
      optimizedGraph.addEdge(logicalNodeEnds[u], logicalNodeStarts[v]);
    }
  }

  // Add nodes and edges inside logical nodes
  for (int i = 0; i < input.nodes.size(); i++) {
    auto &logicalNode = input.nodes[i];

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

    for (auto u : logicalNode.nodes) {
      if (!cudaGraphNodeHasIncomingEdgeMap[u]) {
        optimizedGraph.addEdge(logicalNodeStarts[i], cudaGraphNodeToCustomGraphNodeIdMap[u]);
      }

      if (logicalNode.edges[u].size() == 0) {
        optimizedGraph.addEdge(cudaGraphNodeToCustomGraphNodeIdMap[u], logicalNodeEnds[i]);
      }
    }
  }

  return optimizedGraph;
}
