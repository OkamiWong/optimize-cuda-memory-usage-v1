#include <fmt/core.h>

#include <cstdio>
#include <string>

#include "strategyUtilities.hpp"

inline std::string getLogicalNodeName(OptimizationInput::NodeId index) {
  return fmt::format("logical_node_{}", index);
}

void printEdges(FILE *fp, OptimizationInput &input) {
  for (const auto &[u, destinations] : input.edges) {
    for (auto v : destinations) {
      fmt::print(fp, "{} -> {}\n", getLogicalNodeName(u), getLogicalNodeName(v));
    }
  }
}

void printLogicalNode(FILE *fp, int index, OptimizationInput::LogicalNode &node) {
  fmt::print(fp, "subgraph {} {{\n", getLogicalNodeName(index));
  fmt::print(fp, "label=\"{}\"\n", getLogicalNodeName(index));
  fmt::print(fp, "}}\n");
}

void printLogicalNodes(FILE *fp, OptimizationInput &input) {
  for (int i = 0; i < input.nodes.size(); i++) {
    printLogicalNode(fp, i, input.nodes[i]);
  }
}

void printOptimizationInput(OptimizationInput &input) {
  auto fp = fopen("optimizationInput.dot", "w");

  fmt::print(fp, "digraph G {{\n");

  printLogicalNodes(fp, input);
  printEdges(fp, input);

  fmt::print(fp, "}}\n");

  fclose(fp);
}
