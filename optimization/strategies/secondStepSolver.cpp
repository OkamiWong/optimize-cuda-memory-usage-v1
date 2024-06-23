#include "secondStepSolver.hpp"

#include <fmt/core.h>
#include <ortools/linear_solver/linear_solver.h>

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <numeric>

#include "../../utilities/configurationManager.hpp"
#include "../../utilities/logger.hpp"
#include "../../utilities/utilities.hpp"

using namespace operations_research;

namespace memopt {

// Need separate the or_tools reference with cu files,
// because nvcc cannot compile or_tools
struct IntegerProgrammingSolver {
  static constexpr double totalRunningTimeWeight = 0.0001;
  static constexpr double totalNumberOfDataMovementWeight = 0.00001;

  // States of the solver
  SecondStepSolver::Input input;

  double originalPeakMemoryUsage, lowestPeakMemoryUsagePossible;  // In MiB
  float originalTotalRunningTime;
  double originalPeakMemoryUsageToTotalRunningTimeRatio;
  int numberOfTaskGroups, numberOfArrays, numberOfVertices;
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

  MPVariable *peakMemoryUsage;
  MPVariable *numberOfDataMovements;

  int getTaskGroupVertexIndex(int i) {
    return i * 2 + 1;
  }

  int getTaskGroupStartVertexIndex(int i) {
    return i * 2;
  }

  int getPrefetchVertexIndex(int i, int j) {
    return numberOfTaskGroups * 2 + i * numberOfArrays + j;
  }

  int getOffloadVertexIndex(int i, int j) {
    return numberOfTaskGroups * 2 + numberOfTaskGroups * numberOfArrays + i * numberOfArrays + j;
  }

  void preprocessSecondStepInput() {
    numberOfTaskGroups = input.taskGroupRunningTimes.size();
    numberOfArrays = input.arraySizes.size();
    numberOfVertices = numberOfTaskGroups * 2 + numberOfTaskGroups * numberOfArrays * 2;

    originalTotalRunningTime = input.originalTotalRunningTime;

    lowestPeakMemoryUsagePossible = 0;
    std::set<int> allDependencies, currentDependencies;
    for (int i = 0; i < numberOfTaskGroups; i++) {
      currentDependencies.clear();
      std::set_union(
        input.taskGroupInputArrays[i].begin(),
        input.taskGroupInputArrays[i].end(),
        input.taskGroupOutputArrays[i].begin(),
        input.taskGroupOutputArrays[i].end(),
        std::inserter(currentDependencies, currentDependencies.begin())
      );
      std::set_union(
        currentDependencies.begin(),
        currentDependencies.end(),
        allDependencies.begin(),
        allDependencies.end(),
        std::inserter(allDependencies, allDependencies.begin())
      );
      lowestPeakMemoryUsagePossible = std::max(
        1.0 / 1024.0 / 1024.0 * std::accumulate(currentDependencies.begin(), currentDependencies.end(), static_cast<size_t>(0), [&](size_t a, int b) {
          return a + input.arraySizes[b];
        }),
        lowestPeakMemoryUsagePossible
      );
    }
    originalPeakMemoryUsage = 1.0 / 1024.0 / 1024.0 * std::accumulate(allDependencies.begin(), allDependencies.end(), static_cast<size_t>(0), [&](size_t a, int b) {
                                return a + input.arraySizes[b];
                              });

    originalPeakMemoryUsageToTotalRunningTimeRatio = static_cast<double>(originalPeakMemoryUsage) / static_cast<double>(originalTotalRunningTime);

    shouldAllocate.clear();
    shouldDeallocate.clear();

    // TODO: Uncomment the section below to consider allocation and deallocation
    // in the optimization stage after they are supported in the execution stage.

    // std::vector<int> arrayFirstWritingKernel(numberOfArrays, std::numeric_limits<int>::max());

    // for (auto arr : input.applicationInputArrays) {
    //   arrayFirstWritingKernel[arr] = -1;
    // }

    // for (int i = 0; i < numberOfTaskGroups; i++) {
    //   for (auto arr : input.taskGroupOutputArrays[i]) {
    //     if (i < arrayFirstWritingKernel[arr]) {
    //       arrayFirstWritingKernel[arr] = i;
    //       shouldAllocate[std::make_pair(i, arr)] = true;
    //     }
    //   }
    // }

    // std::vector<int> arrayLastReadingKernel(numberOfArrays, -1);

    // for (auto arr : input.applicationOutputArrays) {
    //   arrayLastReadingKernel[arr] = numberOfTaskGroups;
    // }

    // for (int i = numberOfTaskGroups - 1; i >= 0; i--) {
    //   for (auto arr : input.taskGroupInputArrays[i]) {
    //     if (i > arrayLastReadingKernel[arr]) {
    //       arrayLastReadingKernel[arr] = i;
    //       shouldDeallocate[std::make_pair(i, arr)] = true;
    //     }
    //   }
    // }
  }

  void initialize() {
    solver = std::unique_ptr<MPSolver>(MPSolver::CreateSolver(
      ConfigurationManager::getConfig().optimization.solver
    ));

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
    for (int i = 0; i < numberOfTaskGroups; i++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
      o.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        o[i].push_back({});

        auto sumLessThanOneConstraint = solver->MakeRowConstraint(0, 1);

        sumLessThanOneConstraint->SetCoefficient(p[i][j], 1);

        for (int k = 0; k < numberOfTaskGroups; k++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
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
          for (int v = u + 1; v < numberOfTaskGroups; v++) {
            constraint->SetCoefficient(o[u][j][v], -1);
          }
        }

        auto offloadingConstraint = solver->MakeRowConstraint(0, 1);
        offloadingConstraint->SetCoefficient(y[i][j], 1);
        for (int k = i + 1; k < numberOfTaskGroups; k++) {
          offloadingConstraint->SetCoefficient(o[i][j][k], -1);
        }
      }
    }
  }

  void defineWeights() {
    w.clear();
    w.resize(numberOfVertices);

    // Add weights for kernel vertices
    for (int i = 0; i < numberOfTaskGroups; i++) {
      w[getTaskGroupStartVertexIndex(i)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getTaskGroupStartVertexIndex(i)));
      w[getTaskGroupVertexIndex(i)] = solver->MakeNumVar(
        input.taskGroupRunningTimes[i],
        input.taskGroupRunningTimes[i],
        fmt::format("w_{{{}}}", getTaskGroupVertexIndex(i))
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldDeallocate[std::make_pair(i, j)]) {
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j)));
        } else {
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(0, arrayOffloadingTimes[j], fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j)));

          auto constraint = solver->MakeRowConstraint(0, 0);
          constraint->SetCoefficient(w[getOffloadVertexIndex(i, j)], -1);
          for (int k = 0; k < numberOfTaskGroups; k++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
      e[getTaskGroupStartVertexIndex(i)][getTaskGroupVertexIndex(i)] = oneConstant;
      if (i > 0) {
        e[getTaskGroupVertexIndex(i - 1)][getTaskGroupStartVertexIndex(i)] = oneConstant;
      }
    }

    // Add edges for prefetching vertices
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getTaskGroupStartVertexIndex(i)][getPrefetchVertexIndex(i, j)] = p[i][j];
        for (int k = i; k < numberOfTaskGroups; k++) {
          if (input.taskGroupInputArrays[k].count(j) != 0 || input.taskGroupOutputArrays[k].count(j) != 0) {
            e[getPrefetchVertexIndex(i, j)][getTaskGroupVertexIndex(k)] = oneConstant;
          }
        }
      }
    }

    // Serialize prefetches
    for (int i = 0; i < numberOfTaskGroups; i++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getTaskGroupVertexIndex(i)][getOffloadVertexIndex(i, j)] = solver->MakeBoolVar("");

        auto constraint = solver->MakeRowConstraint(0, 0);
        constraint->SetCoefficient(e[getTaskGroupVertexIndex(i)][getOffloadVertexIndex(i, j)], -1);
        for (int k = 0; k < numberOfTaskGroups; k++) {
          constraint->SetCoefficient(o[i][j][k], 1);
        }

        for (int k = i + 1; k < numberOfTaskGroups; k++) {
          e[getOffloadVertexIndex(i, j)][getTaskGroupStartVertexIndex(k)] = o[i][j][k];
        }
      }
    }

    // Serialize offloadings
    for (int i = 0; i < numberOfTaskGroups; i++) {
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
      if (u == getTaskGroupStartVertexIndex(0)) {
        z.push_back(solver->MakeNumVar(0, 0, ""));
      } else {
        z.push_back(solver->MakeNumVar(0, infinity, ""));
      }
    }

    const float infinityForTime = 10.0 * originalTotalRunningTime;

    for (int u = 0; u < numberOfVertices; u++) {
      if (u != getTaskGroupStartVertexIndex(0)) {
        for (int v = 0; v < numberOfVertices; v++) {
          auto constraint = solver->MakeRowConstraint(-infinityForTime, infinity);
          constraint->SetCoefficient(z[u], 1);
          constraint->SetCoefficient(z[v], -1);
          constraint->SetCoefficient(w[u], -1);
          constraint->SetCoefficient(e[v][u], -infinityForTime);
        }
      }
    }

    auto zLastKernelConstraint = solver->MakeRowConstraint(0, originalTotalRunningTime * ConfigurationManager::getConfig().optimization.acceptableRunningTimeFactor);
    zLastKernelConstraint->SetCoefficient(z[getTaskGroupVertexIndex(numberOfTaskGroups - 1)], 1);
  }

  void addKernelDataDependencyConstraints() {
    std::set<int> nodeInputOutputUnion;
    for (int i = 0; i < numberOfTaskGroups; i++) {
      nodeInputOutputUnion.clear();
      std::set_union(
        input.taskGroupInputArrays[i].begin(),
        input.taskGroupInputArrays[i].end(),
        input.taskGroupOutputArrays[i].begin(),
        input.taskGroupOutputArrays[i].end(),
        std::inserter(nodeInputOutputUnion, nodeInputOutputUnion.begin())
      );
      for (auto &arr : nodeInputOutputUnion) {
        auto constraint = solver->MakeRowConstraint(1, 1);
        constraint->SetCoefficient(y[i][arr], 1);
      }
    }
  }

  void definePeakMemoryUsage() {
    // Represent the peak memory usage in MiB
    peakMemoryUsage = solver->MakeNumVar(0, infinity, "");
    for (int i = 0; i < numberOfTaskGroups; i++) {
      auto constraint = solver->MakeRowConstraint(0, infinity);
      constraint->SetCoefficient(peakMemoryUsage, 1);
      for (int j = 0; j < numberOfArrays; j++) {
        constraint->SetCoefficient(x[i][j], -static_cast<double>(this->input.arraySizes[j]) / 1024.0 / 1024.0);
      }
    }
  }

  void defineNumberOfDataMovements() {
    numberOfDataMovements = solver->MakeNumVar(0, infinity, "");
    auto constraint = solver->MakeRowConstraint(0, infinity);
    constraint->SetCoefficient(numberOfDataMovements, 1);

    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        constraint->SetCoefficient(p[i][j], -1);
      }
    }

    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        for (int k = i + 1; k < numberOfTaskGroups; k++) {
          constraint->SetCoefficient(o[i][j][k], -1);
        }
      }
    }
  }

  void printSolution(MPSolver::ResultStatus resultStatus) {
    LOG_TRACE_WITH_INFO("Printing solution to secondStepSolver.out");

    auto fp = fopen("secondStepSolver.out", "w");

    fmt::print(fp, "Result status: {}\n", resultStatus == MPSolver::OPTIMAL ? "OPTIMAL" : "FEASIBLE");

    auto optimizedPeakMemoryUsage = peakMemoryUsage->solution_value();
    auto totalRunningTime = z[getTaskGroupVertexIndex(numberOfTaskGroups - 1)]->solution_value();

    fmt::print(fp, "Original peak memory usage (MiB): {:.6f}\n", originalPeakMemoryUsage);
    fmt::print(fp, "Lowest peak memory usage possible (MiB): {:.6f}\n", lowestPeakMemoryUsagePossible);
    fmt::print(fp, "Optimal peak memory usage (MiB): {:.6f}\n", optimizedPeakMemoryUsage);
    fmt::print(fp, "Optimal peak memory usage  / Original peak memory usage: {:.6f}%\n", optimizedPeakMemoryUsage / originalPeakMemoryUsage * 100.0);

    fmt::print(fp, "Original total running time (s): {:.6f}\n", originalTotalRunningTime);
    fmt::print(fp, "Total running time (s): {:.6f}\n", totalRunningTime);
    fmt::print(fp, "Total running time / original: {:.6f}%\n", totalRunningTime / originalTotalRunningTime * 100.0);

    fmt::print(fp, "Solution:\n");

    for (int i = 0; i < numberOfTaskGroups; i++) {
      fmt::print(fp, "{}:\n", i);

      fmt::print(fp, "  Inputs: ");
      for (auto arrayIndex : input.taskGroupInputArrays[i]) {
        fmt::print(fp, "{}, ", arrayIndex);
      }
      fmt::print(fp, "\n");

      fmt::print(fp, "  Outputs: ");
      for (auto arrayIndex : input.taskGroupOutputArrays[i]) {
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

    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (p[i][j]->solution_value() > 0) {
          fmt::print(fp, "p_{{{}, {}}} = {}; w = {}; z = {}\n", i, j, true, w[getPrefetchVertexIndex(i, j)]->solution_value(), z[getPrefetchVertexIndex(i, j)]->solution_value());
        }
      }
    }

    fmt::print(fp, "\n");

    for (int i = 0; i < numberOfTaskGroups; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        for (int k = 0; k < numberOfTaskGroups; k++) {
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
    for (int i = 0; i < numberOfTaskGroups; i++) {
      auto duration = input.taskGroupRunningTimes[i];
      minDuration = std::min(minDuration, duration);
      maxDuration = std::max(maxDuration, duration);
      fmt::print(fp, "taskGroupRunningTimes[{}] = {}\n", i, duration);
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

    fclose(fp);
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

    definePeakMemoryUsage();

    defineNumberOfDataMovements();

    auto objective = solver->MutableObjective();
    objective->SetCoefficient(peakMemoryUsage, 1);
    objective->SetCoefficient(numberOfDataMovements, totalNumberOfDataMovementWeight);
    objective->SetCoefficient(
      z[getTaskGroupVertexIndex(numberOfTaskGroups - 1)],
      originalPeakMemoryUsageToTotalRunningTimeRatio * totalRunningTimeWeight
    );
    objective->SetMinimization();

    solver->set_time_limit(1000 * 60 * 30);

    MPSolverParameters solverParam;
    // solverParam.SetIntegerParam(
    //   MPSolverParameters::IntegerParam::LP_ALGORITHM,
    //   MPSolverParameters::LpAlgorithmValues::PRIMAL
    // );

    SystemWallClock clock;

    clock.start();
    auto resultStatus = solver->Solve(solverParam);
    clock.end();

    LOG_TRACE_WITH_INFO("Time for solving the MIP problem (seconds): %.2f", clock.getTimeInSeconds());

    SecondStepSolver::Output output;

    if (resultStatus == MPSolver::OPTIMAL || resultStatus == MPSolver::FEASIBLE) {
      printSolution(resultStatus);

      output.optimal = true;

      for (int i = 0; i < numberOfArrays; i++) {
        if (initiallyAllocatedOnDevice[i]->solution_value() > 0) {
          output.indicesOfArraysInitiallyOnDevice.push_back(i);
        }
      }

      for (int i = 0; i < numberOfTaskGroups; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          if (p[i][j]->solution_value() > 0) {
            output.prefetches.push_back(std::make_tuple(i, j));
          }
        }
      }

      for (int i = 0; i < numberOfTaskGroups; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          for (int k = 0; k < numberOfTaskGroups; k++) {
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

}  // namespace memopt
