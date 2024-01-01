#pragma once

#include <vector>
#include "../optimizationInput.hpp"

class FirstStepSolver {
 public:
  struct Input {
    std::vector<std::vector<size_t>> dataDependencyOverlapInBytes;
    std::vector<std::vector<bool>> canPrecedeInTopologicalSort;
  };

  struct Output {
  };

  static Input constructInput(OptimizationInput optimizationInput);
};

class SecondStepSolver {
 public:
  struct Input {
  };

  struct Output {
  };

  static Input constructInput();
};
