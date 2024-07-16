#include <fmt/core.h>

#include <cstdio>
#include <string>

#include "../../utilities/logger.hpp"
#include "strategyUtilities.hpp"

namespace memopt {

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
  std::string outputFilePath = fmt::format("debug/{}.optimizationInput.dot", input.stageIndex);
  LOG_TRACE_WITH_INFO("Printing OptimizationInput to %s", outputFilePath.c_str());

  auto fp = fopen(outputFilePath.c_str(), "w");

  fmt::print(fp, "digraph G {{\n");

  printTaskGroups(fp, input);
  printEdges(fp, input);

  fmt::print(fp, "}}\n");

  fclose(fp);
}

inline std::string getNodeName(OptimizationOutput &output, int u) {
  return fmt::format("node_{}", u);
}

void printNodes(FILE *fp, OptimizationOutput &output) {
  for (auto u : output.nodes) {
    std::string nodeDescription;
    if (output.nodeIdToNodeTypeMap[u] == OptimizationOutput::NodeType::dataMovement) {
      nodeDescription = "type=dataMovement";
    } else if (output.nodeIdToNodeTypeMap[u] == OptimizationOutput::NodeType::task) {
      nodeDescription = fmt::format("type=task, taskId={}", output.nodeIdToTaskIdMap[u]);
    } else if (output.nodeIdToNodeTypeMap[u] == OptimizationOutput::NodeType::empty) {
      nodeDescription = "type=empty";
    } else {
      nodeDescription = "type=UNKNOWN";
    }

    fmt::print(fp, "{} [label=\"{} {}\"]\n", getNodeName(output, u), getNodeName(output, u), nodeDescription);
  }
}

void printEdges(FILE *fp, OptimizationOutput &output) {
  for (const auto &[u, destinations] : output.edges) {
    for (auto v : destinations) {
      fmt::print(fp, "{} -> {}\n", getNodeName(output, u), getNodeName(output, v));
    }
  }
}

void printOptimizationOutput(OptimizationOutput &output, int stageIndex) {
  std::string outputFilePath = fmt::format("debug/{}.optimizationOutput.dot", stageIndex);
  LOG_TRACE_WITH_INFO("Printing OptimizationOutput to %s", outputFilePath.c_str());

  auto fp = fopen(outputFilePath.c_str(), "w");

  fmt::print(fp, "digraph G {{\n");

  printNodes(fp, output);
  printEdges(fp, output);

  fmt::print(fp, "}}\n");

  fclose(fp);
}

}  // namespace memopt
