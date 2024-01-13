#include "secondStepSolver.hpp"

#include <fmt/core.h>
#include <ortools/linear_solver/linear_solver.h>

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <numeric>

#include "../../utilities/constants.hpp"
#include "../../utilities/logger.hpp"

using namespace operations_research;

// Need separate the or_tools reference with cu files,
// because nvcc cannot compile or_tools
struct IntegerProgrammingSolver {
  // States of the solver
  SecondStepSolver::Input input;
  size_t originalPeakMemoryUsage;
  float originalTotalTime;
  int numberOfLogicalNodes, numberOfArrays, numberOfVertices;
  std::map<std::pair<int, int>, bool> shouldAllocate, shouldDeallocate;

  std::unique_ptr<MPSolver> solver;
  double infinity;

  // Variables of the integer programming problem
  std::vector<MPVariable *> initiallyAllocatedOnDevice;
  std::vector<std::vector<MPVariable *>> p;
  std::vector<std::vector<std::vector<MPVariable *>>> o;

  std::vector<std::vector<MPVariable *>> x;
  std::vector<std::vector<MPVariable *>> y;

  std::vector<MPVariable *> w;
  std::vector<double> arrayPrefetchingTimes, arrayOffloadingTimes;

  std::vector<std::vector<MPVariable *>> e;

  std::vector<MPVariable *> z;

  int getLogicalNodeVertexIndex(int i) {
    return i * 2 + 1;
  }

  int getLogicalNodeStartVertexIndex(int i) {
    return i * 2;
  }

  int getPrefetchVertexIndex(int i, int j) {
    return numberOfLogicalNodes * 2 + i * numberOfArrays + j;
  }

  int getOffloadVertexIndex(int i, int j) {
    return numberOfLogicalNodes * 2 + numberOfLogicalNodes * numberOfArrays + i * numberOfArrays + j;
  }

  void preprocessSecondStepInput() {
    numberOfLogicalNodes = input.nodeDurations.size();
    numberOfArrays = input.arraySizes.size();
    numberOfVertices = numberOfLogicalNodes * 2 + numberOfLogicalNodes * numberOfArrays * 2;

    originalTotalTime = std::accumulate(
      input.nodeDurations.begin(),
      input.nodeDurations.end(),
      0.0f,
      [](float a, float b) {
        return a + b;
      }
    );

    std::set<int> allDependencies, currentDependencies;
    originalPeakMemoryUsage = 0;
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      currentDependencies.clear();
      std::set_union(
        input.nodeInputArrays[i].begin(),
        input.nodeInputArrays[i].end(),
        input.nodeOutputArrays[i].begin(),
        input.nodeOutputArrays[i].end(),
        std::inserter(currentDependencies, currentDependencies.begin())
      );
      std::set_union(
        currentDependencies.begin(),
        currentDependencies.end(),
        allDependencies.begin(),
        allDependencies.end(),
        std::inserter(allDependencies, allDependencies.begin())
      );
    }
    originalPeakMemoryUsage = std::accumulate(
      allDependencies.begin(),
      allDependencies.end(),
      static_cast<size_t>(0),
      [&](size_t a, int b) {
        return a + input.arraySizes[b];
      }
    );

    shouldAllocate.clear();
    std::vector<int> arrayFirstWritingKernel(numberOfArrays, std::numeric_limits<int>::max());

    for (auto arr : input.applicationInputArrays) {
      arrayFirstWritingKernel[arr] = -1;
    }

    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (auto arr : input.nodeOutputArrays[i]) {
        if (i < arrayFirstWritingKernel[arr]) {
          arrayFirstWritingKernel[arr] = i;
          shouldAllocate[std::make_pair(i, arr)] = true;
        }
      }
    }

    shouldDeallocate.clear();
    std::vector<int> arrayLastReadingKernel(numberOfArrays, -1);

    for (auto arr : input.applicationOutputArrays) {
      arrayLastReadingKernel[arr] = numberOfLogicalNodes;
    }

    for (int i = numberOfLogicalNodes - 1; i >= 0; i--) {
      for (auto arr : input.nodeInputArrays[i]) {
        if (i > arrayLastReadingKernel[arr]) {
          arrayLastReadingKernel[arr] = i;
          shouldDeallocate[std::make_pair(i, arr)] = true;
        }
      }
    }
  }

  void initialize() {
    solver = std::unique_ptr<MPSolver>(MPSolver::CreateSolver("SCIP"));

    if (!solver) {
      fmt::print("Solver not available\n");
    }

    infinity = solver->infinity();
  }

  void defineDecisionVariables() {
    // Add decision variables for initially allocated on device or not
    initiallyAllocatedOnDevice.clear();
    for (int i = 0; i < numberOfArrays; i++) {
      initiallyAllocatedOnDevice.push_back(solver->MakeBoolVar(fmt::format("I_{{{}}}", i)));
      initiallyAllocatedOnDevice[i]->SetUB(0);
    }
    for (auto arr : input.applicationInputArrays) {
      initiallyAllocatedOnDevice[arr]->SetUB(1);
    }

    // Add decision variables for prefetching
    p.clear();
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      p.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        p[i].push_back(solver->MakeBoolVar(fmt::format("p_{{{}, {}}}", i, j)));
        if (shouldAllocate[std::make_pair(i, j)]) {
          p[i][j]->SetLB(1);
        }
      }
    }

    // Add decision variables for offloading
    o.clear();
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      o.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        o[i].push_back({});

        auto sumLessThanOneConstraint = solver->MakeRowConstraint(0, 1);

        sumLessThanOneConstraint->SetCoefficient(p[i][j], 1);

        for (int k = 0; k < numberOfLogicalNodes; k++) {
          o[i][j].push_back(solver->MakeBoolVar(fmt::format("o_{{{},{},{}}}", i, j, k)));

          sumLessThanOneConstraint->SetCoefficient(o[i][j][k], 1);

          if (shouldDeallocate[std::make_pair(i, j)]) {
            if (k == i + 1) {
              o[i][j][k]->SetLB(1);
            } else {
              o[i][j][k]->SetUB(0);
            }
          } else {
            if (k <= i) {
              o[i][j][k]->SetUB(0);
            }
          }
        }
      }
    }
  }

  void defineXAndY() {
    // Define the variables representing whether the memory for an array is allocated on the device
    x.clear();
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      x.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        x[i].push_back(solver->MakeBoolVar(fmt::format("x_{{{}, {}}}", i, j)));

        auto constraint = solver->MakeRowConstraint(0, 0);
        constraint->SetCoefficient(x[i][j], -1);
        constraint->SetCoefficient(initiallyAllocatedOnDevice[j], 1);
        for (int u = 0; u <= i; u++) {
          constraint->SetCoefficient(p[u][j], 1);
        }
        for (int u = 0; u <= i - 1; u++) {
          for (int v = u + 1; v <= i; v++) {
            constraint->SetCoefficient(o[u][j][v], -1);
          }
        }
      }
    }

    // Define the variables representing whether an array is available
    y.clear();
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      y.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        y[i].push_back(solver->MakeBoolVar(fmt::format("y_{{{}, {}}}", i, j)));

        auto constraint = solver->MakeRowConstraint(0, 0);
        constraint->SetCoefficient(y[i][j], -1);
        constraint->SetCoefficient(initiallyAllocatedOnDevice[j], 1);
        for (int u = 0; u <= i; u++) {
          constraint->SetCoefficient(p[u][j], 1);
        }
        for (int u = 0; u <= i - 1; u++) {
          for (int v = u + 1; v < numberOfLogicalNodes; v++) {
            constraint->SetCoefficient(o[u][j][v], -1);
          }
        }

        auto offloadingConstraint = solver->MakeRowConstraint(0, 1);
        offloadingConstraint->SetCoefficient(y[i][j], 1);
        for (int k = i + 1; k < numberOfLogicalNodes; k++) {
          offloadingConstraint->SetCoefficient(o[i][j][k], -1);
        }
      }
    }
  }

  void defineWeights() {
    w.clear();
    w.resize(numberOfVertices);

    // Add weights for kernel vertices
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      w[getLogicalNodeStartVertexIndex(i)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getLogicalNodeStartVertexIndex(i)));
      w[getLogicalNodeVertexIndex(i)] = solver->MakeNumVar(
        input.nodeDurations[i],
        input.nodeDurations[i],
        fmt::format("w_{{{}}}", getLogicalNodeVertexIndex(i))
      );
    }

    // Preprocess weights for data movement vertices
    arrayPrefetchingTimes.clear();
    arrayOffloadingTimes.clear();
    for (int i = 0; i < numberOfArrays; i++) {
      arrayPrefetchingTimes.push_back(input.arraySizes[i] / input.prefetchingBandwidth);
      arrayOffloadingTimes.push_back(input.arraySizes[i] / input.offloadingBandwidth);
    }

    // Add weights for prefetches
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldAllocate[std::make_pair(i, j)]) {
          w[getPrefetchVertexIndex(i, j)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getPrefetchVertexIndex(i, j)));
        } else {
          w[getPrefetchVertexIndex(i, j)] = solver->MakeNumVar(0, arrayPrefetchingTimes[j], fmt::format("w_{{{}}}", getPrefetchVertexIndex(i, j)));
          auto constraint = solver->MakeRowConstraint(0, 0);
          constraint->SetCoefficient(w[getPrefetchVertexIndex(i, j)], -1);
          constraint->SetCoefficient(p[i][j], arrayPrefetchingTimes[j]);
        }
      }
    }

    // Add weights for offloadings
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldDeallocate[std::make_pair(i, j)]) {
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j)));
        } else {
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(0, arrayOffloadingTimes[j], fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j)));

          auto constraint = solver->MakeRowConstraint(0, 0);
          constraint->SetCoefficient(w[getOffloadVertexIndex(i, j)], -1);
          for (int k = 0; k < numberOfLogicalNodes; k++) {
            constraint->SetCoefficient(o[i][j][k], arrayOffloadingTimes[j]);
          }
        }
      }
    }
  }

  void addEdges() {
    auto zeroConstant = solver->MakeIntVar(0, 0, "zero");
    auto oneConstant = solver->MakeIntVar(1, 1, "one");

    e.clear();
    e.resize(numberOfVertices, std::vector<MPVariable *>(numberOfVertices, zeroConstant));

    // Add edges between kernel and kernel start vertices
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      e[getLogicalNodeStartVertexIndex(i)][getLogicalNodeVertexIndex(i)] = oneConstant;
      if (i > 0) {
        e[getLogicalNodeVertexIndex(i - 1)][getLogicalNodeStartVertexIndex(i)] = oneConstant;
      }
    }

    // Add edges for prefetching vertices
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getLogicalNodeStartVertexIndex(i)][getPrefetchVertexIndex(i, j)] = p[i][j];
        for (int k = i; k < numberOfLogicalNodes; k++) {
          if (input.nodeInputArrays[k].count(j) != 0 || input.nodeOutputArrays[k].count(j) != 0) {
            e[getPrefetchVertexIndex(i, j)][getLogicalNodeVertexIndex(k)] = oneConstant;
          }
        }
      }
    }

    // Serialize prefetches
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (j == 0) {
          if (i != 0) {
            e[getPrefetchVertexIndex(i - 1, numberOfArrays - 1)][getPrefetchVertexIndex(i, j)] = oneConstant;
          }
        } else {
          e[getPrefetchVertexIndex(i, j - 1)][getPrefetchVertexIndex(i, j)] = oneConstant;
        }
      }
    }

    // Add edges for offloading vertices
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getLogicalNodeVertexIndex(i)][getOffloadVertexIndex(i, j)] = solver->MakeBoolVar("");

        auto constraint = solver->MakeRowConstraint(0, 0);
        constraint->SetCoefficient(e[getLogicalNodeVertexIndex(i)][getOffloadVertexIndex(i, j)], -1);
        for (int k = 0; k < numberOfLogicalNodes; k++) {
          constraint->SetCoefficient(o[i][j][k], 1);
        }

        for (int k = i + 1; k < numberOfLogicalNodes; k++) {
          e[getOffloadVertexIndex(i, j)][getLogicalNodeStartVertexIndex(k)] = o[i][j][k];
        }
      }
    }

    // Serialize offloadings
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (j == 0) {
          if (i != 0) {
            e[getOffloadVertexIndex(i - 1, numberOfArrays - 1)][getOffloadVertexIndex(i, j)] = oneConstant;
          }
        } else {
          e[getOffloadVertexIndex(i, j - 1)][getOffloadVertexIndex(i, j)] = oneConstant;
        }
      }
    }
  }

  void addLongestPathConstraints() {
    z.clear();

    for (int u = 0; u < numberOfVertices; u++) {
      if (u == getLogicalNodeStartVertexIndex(0)) {
        z.push_back(solver->MakeNumVar(0, 0, ""));
      } else {
        z.push_back(solver->MakeNumVar(0, infinity, ""));
      }
    }

    const float infinityForTime = 10.0 * originalTotalTime;

    for (int u = 0; u < numberOfVertices; u++) {
      if (u != getLogicalNodeStartVertexIndex(0)) {
        for (int v = 0; v < numberOfVertices; v++) {
          auto constraint = solver->MakeRowConstraint(-infinityForTime, infinity);
          constraint->SetCoefficient(z[u], 1);
          constraint->SetCoefficient(z[v], -1);
          constraint->SetCoefficient(w[u], -1);
          constraint->SetCoefficient(e[v][u], -infinityForTime);
        }
      }
    }

    auto zLastKernelConstraint = solver->MakeRowConstraint(0, originalTotalTime * Constants::ACCEPTABLE_RUNNING_TIME_FACTOR);
    zLastKernelConstraint->SetCoefficient(z[getLogicalNodeVertexIndex(numberOfLogicalNodes - 1)], 1);
  }

  void addKernelDataDependencyConstraints() {
    std::set<int> nodeInputOutputUnion;
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      nodeInputOutputUnion.clear();
      std::set_union(
        input.nodeInputArrays[i].begin(),
        input.nodeInputArrays[i].end(),
        input.nodeOutputArrays[i].begin(),
        input.nodeOutputArrays[i].end(),
        std::inserter(nodeInputOutputUnion, nodeInputOutputUnion.begin())
      );
      for (auto &arr : nodeInputOutputUnion) {
        auto constraint = solver->MakeRowConstraint(1, 1);
        constraint->SetCoefficient(y[i][arr], 1);
      }
    }
  }

  void printSolution(MPObjective *objective) {
    LOG_TRACE_WITH_INFO("Printing solution to secondStepSolver.out");

    auto fp = fopen("secondStepSolver.out", "w");

    auto optimizedPeakMemoryUsage = objective->Value();
    auto totalRunningTime = z[getLogicalNodeVertexIndex(numberOfLogicalNodes - 1)]->solution_value();

    fmt::print(fp, "Original peak memory usage (MByte): {:.6f}\n", originalPeakMemoryUsage * 1e-6);
    fmt::print(fp, "Optimal peak memory usage (MByte): {:.6f}\n", optimizedPeakMemoryUsage);
    fmt::print(fp, "Optimal peak memory usage  / Original peak memory usage: {:.6f}%\n", optimizedPeakMemoryUsage * 1e6 / originalPeakMemoryUsage * 100.0);

    fmt::print(fp, "Original total running time (s): {:.6f}\n", originalTotalTime);
    fmt::print(fp, "Total running time (s): {:.6f}\n", totalRunningTime);
    fmt::print(fp, "Total running time / original: {:.6f}%\n", totalRunningTime / originalTotalTime * 100.0);

    fmt::print(fp, "Solution:\n");

    for (int i = 0; i < numberOfLogicalNodes; i++) {
      fmt::print(fp, "{}:\n", i);

      fmt::print(fp, "  Inputs: ");
      for (auto arrayIndex : input.nodeInputArrays[i]) {
        fmt::print(fp, "{}, ", arrayIndex);
      }
      fmt::print(fp, "\n");

      fmt::print(fp, "  Outputs: ");
      for (auto arrayIndex : input.nodeOutputArrays[i]) {
        fmt::print(fp, "{}, ", arrayIndex);
      }
      fmt::print(fp, "\n");
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfArrays; i++) {
      if (initiallyAllocatedOnDevice[i]->solution_value() > 0) {
        fmt::print(fp, "I_{{{}}} = 1\n", i);
      }
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (p[i][j]->solution_value() > 0) {
          fmt::print(fp, "p_{{{}, {}}} = {}; w = {}; z = {}\n", i, j, true, w[getPrefetchVertexIndex(i, j)]->solution_value(), z[getPrefetchVertexIndex(i, j)]->solution_value());
        }
      }
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfLogicalNodes; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        for (int k = 0; k < numberOfLogicalNodes; k++) {
          if (o[i][j][k]->solution_value() > 0) {
            fmt::print(fp, "o_{{{}, {}, {}}} = {}; w = {}; z = {}\n", i, j, k, true, w[getOffloadVertexIndex(i, j)]->solution_value(), z[getOffloadVertexIndex(i, j)]->solution_value());
          }
        }
      }
    }

    fclose(fp);
  }

  // For checking scale of numbers
  void printDurationsAndSizes() {
    LOG_TRACE_WITH_INFO("Printing input to secondStepSolverInput.out");

    auto fp = fopen("secondStepSolverInput.out", "w");
    float minDuration = std::numeric_limits<float>::max();
    float maxDuration = 0;
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      auto duration = input.nodeDurations[i];
      minDuration = std::min(minDuration, duration);
      maxDuration = std::max(maxDuration, duration);
      fmt::print(fp, "nodeDurations[{}] = {}\n", i, duration);
    }

    size_t minSize = std::numeric_limits<size_t>::max();
    size_t maxSize = 0;
    for (int i = 0; i < numberOfArrays; i++) {
      auto size = input.arraySizes[i];
      minSize = std::min(minSize, size);
      maxSize = std::max(maxSize, size);
      fmt::print(fp, "arraySizes[{}] = {}\n", i, size);
    }

    fmt::print(
      fp,
      "minDuration = {}\nmaxDuration = {}\nminSize = {}\nmaxSize = {}\n",
      minDuration,
      maxDuration,
      minSize,
      maxSize
    );
  }

  SecondStepSolver::Output solve(SecondStepSolver::Input &&input, bool verbose = false) {
    this->input = std::move(input);

    initialize();

    preprocessSecondStepInput();

    printDurationsAndSizes();

    defineDecisionVariables();

    defineXAndY();

    defineWeights();

    addEdges();

    addLongestPathConstraints();

    addKernelDataDependencyConstraints();

    // Represent the peak memory usage in MByte
    auto peakMemoryUsage = solver->MakeNumVar(0, infinity, "");
    for (int i = 0; i < numberOfLogicalNodes; i++) {
      auto constraint = solver->MakeRowConstraint(0, infinity);
      constraint->SetCoefficient(peakMemoryUsage, 1);
      for (int j = 0; j < numberOfArrays; j++) {
        constraint->SetCoefficient(x[i][j], -static_cast<double>(this->input.arraySizes[j]) * 1e-6);
      }
    }

    auto objective = solver->MutableObjective();
    objective->SetCoefficient(peakMemoryUsage, 1);
    objective->SetMinimization();

    solver->set_time_limit(1000 * 30);

    auto resultStatus = solver->Solve();

    SecondStepSolver::Output output;

    if (resultStatus == MPSolver::OPTIMAL) {
      printSolution(objective);

      output.optimal = true;

      for (int i = 0; i < numberOfArrays; i++) {
        if (initiallyAllocatedOnDevice[i]->solution_value() > 0) {
          output.indicesOfArraysInitiallyOnDevice.push_back(i);
        }
      }

      for (int i = 0; i < numberOfLogicalNodes; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          if (p[i][j]->solution_value() > 0) {
            output.prefetches.push_back(std::make_tuple(i, j));
          }
        }
      }

      for (int i = 0; i < numberOfLogicalNodes; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          for (int k = 0; k < numberOfLogicalNodes; k++) {
            if (o[i][j][k]->solution_value() > 0) {
              output.offloadings.push_back(std::make_tuple(i, j, k));
            }
          }
        }
      }
    } else {
      LOG_TRACE_WITH_INFO("No optimal solution found. (ResultStatus=%d)", resultStatus);
      output.optimal = false;
    }

    return output;
  }
};

SecondStepSolver::Output SecondStepSolver::solve(SecondStepSolver::Input &&input) {
  LOG_TRACE();

  IntegerProgrammingSolver solver;
  return solver.solve(std::move(input));
}
