#pragma once

#include <vector>

#include "../../utilities/types.hpp"

namespace memopt {

class FirstStepSolver {
 public:
  struct Input {
    TaskGroupId n;
    std::vector<std::vector<TaskGroupId>> edges;
    std::vector<std::vector<size_t>> dataDependencyOverlapInBytes;
    int stageIndex;
  };

  struct Output {
    std::vector<TaskGroupId> taskGroupExecutionOrder;
  };

  FirstStepSolver(Input &&input);
  Output solve();

 private:
  Input input;
  Output output;
  size_t maxTotalOverlap;
  std::vector<bool> visited;
  std::vector<int> inDegree;
  std::vector<TaskGroupId> currentTopologicalSort;

  void dfs(size_t currentTotalOverlap);
  void printSolution();
};

}  // namespace memopt
