#pragma once

#include <vector>

class FirstStepSolver {
 public:
  struct Input {
    int n;
    std::vector<std::vector<size_t>> dataDependencyOverlapInBytes;
    std::vector<std::vector<bool>> canPrecedeInTopologicalSort;
  };

  struct Output {
    std::vector<int> nodeExecutionOrder;
  };

  FirstStepSolver(Input &&input);
  Output solve();

 private:
  Input input;
};
