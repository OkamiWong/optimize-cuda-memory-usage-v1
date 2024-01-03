#pragma once

#include <vector>

class FirstStepSolver {
 public:
  struct Input {
    int n;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<size_t>> dataDependencyOverlapInBytes;
  };

  struct Output {
    std::vector<int> nodeExecutionOrder;
  };

  FirstStepSolver(Input &&input);
  Output solve();

 private:
  Input input;
  Output output;
  size_t maxTotalOverlap;
  std::vector<bool> visited;
  std::vector<int> inDegree;
  std::vector<int> currentTopologicalSort;

  void dfs(size_t currentTotalOverlap);
};
