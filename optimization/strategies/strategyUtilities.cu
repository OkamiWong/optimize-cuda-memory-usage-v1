#include <fmt/core.h>

#include <string>

#include "strategyUtilities.hpp"

inline std::string getLogicalNodeName(OptimizationInput::NodeId index) {
  return fmt::format("logical_node_{}", index);
}

void printEdges(OptimizationInput &input) {
  for (const auto &[u, destinations] : input.edges) {
    for (auto v : destinations) {
      fmt::print("{} -> {}\n", getLogicalNodeName(u), getLogicalNodeName(v));
    }
  }
}

void printLogicalNode(int index, OptimizationInput::LogicalNode &node) {
  fmt::print("subgraph {} {{\n", getLogicalNodeName(index));
  fmt::print("label=\"{}\"\n", getLogicalNodeName(index));
  fmt::print("}}\n");
}

void printLogicalNodes(OptimizationInput &input) {
  for (int i = 0; i < input.nodes.size(); i++) {
    printLogicalNode(i, input.nodes[i]);
  }
}

void printOptimizationInput(OptimizationInput &input) {
  fmt::print("graph G {{\n");

  printLogicalNodes(input);

  printEdges(input);

  fmt::print("}}\n");
}
