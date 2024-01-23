#include <fmt/core.h>

#include <cstdio>
#include <string>

#include "../../utilities/logger.hpp"
#include "strategyUtilities.hpp"

inline std::string getTaskGroupName(TaskGroupId index) {
  return fmt::format("task_group_{}", index);
}

void printEdges(FILE *fp, OptimizationInput &input) {
  for (const auto &[u, destinations] : input.edges) {
    for (auto v : destinations) {
      fmt::print(fp, "{} -> {}\n", getTaskGroupName(u), getTaskGroupName(v));
    }
  }
}

void printTaskGroup(FILE *fp, int index, OptimizationInput::TaskGroup &taskGroup) {
  fmt::print(fp, "subgraph {} {{\n", getTaskGroupName(index));
  fmt::print(fp, "label=\"{} (size={})\"\n", getTaskGroupName(index), taskGroup.nodes.size());
  fmt::print(fp, "}}\n");
}

void printTaskGroups(FILE *fp, OptimizationInput &input) {
  for (int i = 0; i < input.nodes.size(); i++) {
    printTaskGroup(fp, i, input.nodes[i]);
  }
}

void printOptimizationInput(OptimizationInput &input) {
  LOG_TRACE_WITH_INFO("Printing OptimizationInput to optimizationInput.dot");

  auto fp = fopen("optimizationInput.dot", "w");

  fmt::print(fp, "digraph G {{\n");

  printTaskGroups(fp, input);
  printEdges(fp, input);

  fmt::print(fp, "}}\n");

  fclose(fp);
}
