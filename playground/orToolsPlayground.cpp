#include <fmt/core.h>
#include <ortools/linear_solver/linear_solver.h>

using namespace operations_research;

namespace operations_research {
void BasicExample() {
  // Create the linear solver with the GLOP backend.
  std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));

  // Create the variables x and y.
  MPVariable *const x = solver->MakeNumVar(0.0, 1, "x");
  MPVariable *const y = solver->MakeNumVar(0.0, 2, "y");
  auto z = solver->MakeBoolVar("z");

  LOG(INFO) << "Number of variables = " << solver->NumVariables();

  // Create a linear constraint, 0 <= x + y <= 2.
  MPConstraint *const ct = solver->MakeRowConstraint(0.0, 2.0, "ct");
  ct->SetCoefficient(x, 1);
  ct->SetCoefficient(y, 1);

  LOG(INFO) << "Number of constraints = " << solver->NumConstraints();

  // Create the objective function, 3 * x + y.
  MPObjective *const objective = solver->MutableObjective();
  objective->SetCoefficient(x, 3);
  objective->SetCoefficient(y, 1);
  objective->SetCoefficient(z, 1);
  objective->SetMaximization();

  solver->Solve();

  LOG(INFO) << "Solution:" << std::endl;
  LOG(INFO) << "Objective value = " << objective->Value();
  LOG(INFO) << "x = " << x->solution_value();
  LOG(INFO) << "y = " << y->solution_value();
  LOG(INFO) << "z = " << z->solution_value();
}
}  // namespace operations_research

struct TwoStepIntegerProgrammingStrategy {
  // Experiment setup
  static constexpr int NUMBER_OF_KERNELS = 4;
  static constexpr double KERNEL_RUNNING_TIME = 1;
  static constexpr double CONNECTION_BANDWIDTH = 281.0 * 1e9;
  static constexpr size_t ARRAY_SIZE = CONNECTION_BANDWIDTH * KERNEL_RUNNING_TIME;
  static constexpr double ACCEPTABLE_RUNNING_TIME_FACTOR = 1.0;

  struct SecondStepInput {
    std::vector<int> kernelExecutionSequence;
    std::vector<double> kernelRunningTimes;

    std::vector<size_t> arraySizes;
    std::set<int> applicationInputArrays, applicationOutputArrays;
    std::vector<std::set<int>> kernelInputArrays, kernelOutputArrays;

    double prefetchingBandwidth, offloadingBandwidth;
  };

  static SecondStepInput getExampleSecondStepInput() {
    SecondStepInput secondStepInput;

    for (int i = 0; i < NUMBER_OF_KERNELS; i++) {
      secondStepInput.kernelExecutionSequence.push_back(i);
      secondStepInput.kernelRunningTimes.push_back(KERNEL_RUNNING_TIME);
      secondStepInput.kernelInputArrays.push_back({i * 3, i * 3 + 1});
      secondStepInput.kernelOutputArrays.push_back({i * 3 + 2});
      secondStepInput.applicationInputArrays.insert({i * 3, i * 3 + 1});
      secondStepInput.applicationOutputArrays.insert({i * 3 + 2});
    }

    for (int i = 0; i < NUMBER_OF_KERNELS; i++) {
      for (int j = i * 3; j < (i + 1) * 3; j++) {
        secondStepInput.arraySizes.push_back(ARRAY_SIZE);
      }
    }

    secondStepInput.prefetchingBandwidth = secondStepInput.offloadingBandwidth = CONNECTION_BANDWIDTH;

    return secondStepInput;
  }

  // States of the strategy
  int numberOfKernels, numberOfArrays, numberOfVertices;
  std::map<std::pair<int, int>, bool> shouldAllocate, shouldDeallocate;

  int getKernelVertexIndex(int i) {
    return i * 2 + 1;
  }

  int getKernelStartVertexIndex(int i) {
    return i * 2;
  }

  int getPrefetchVertexIndex(int i, int j) {
    return numberOfKernels * 2 + i * numberOfArrays + j;
  }

  int getOffloadVertexIndex(int i, int j) {
    return numberOfKernels * 2 + numberOfKernels * numberOfArrays + i * numberOfArrays + j;
  }

  void preprocessSecondStepInput(const SecondStepInput &secondStepInput) {
    numberOfKernels = secondStepInput.kernelExecutionSequence.size();
    numberOfArrays = secondStepInput.arraySizes.size();
    numberOfVertices = numberOfKernels * 2 + numberOfKernels * numberOfArrays * 2;

    shouldAllocate.clear();
    std::vector<int> arrayFirstWritingKernel(numberOfArrays, std::numeric_limits<int>::max());

    for (auto arr : secondStepInput.applicationInputArrays) {
      arrayFirstWritingKernel[arr] = -1;
    }

    for (int i = 0; i < numberOfKernels; i++) {
      for (auto arr : secondStepInput.kernelOutputArrays[i]) {
        if (i < arrayFirstWritingKernel[arr]) {
          arrayFirstWritingKernel[arr] = i;
          shouldAllocate[std::make_pair(i, arr)] = true;
        }
      }
    }

    shouldDeallocate.clear();
    std::vector<int> arrayLastReadingKernel(numberOfArrays, -1);

    for (auto arr : secondStepInput.applicationOutputArrays) {
      arrayLastReadingKernel[arr] = numberOfKernels;
    }

    for (int i = numberOfKernels - 1; i >= 0; i--) {
      for (auto arr : secondStepInput.kernelInputArrays[i]) {
        if (i > arrayLastReadingKernel[arr]) {
          arrayLastReadingKernel[arr] = i;
          shouldDeallocate[std::make_pair(i, arr)] = true;
        }
      }
    }
  }

  std::unique_ptr<MPSolver> solver;
  double infinity;

  void initialize() {
    solver = std::unique_ptr<MPSolver>(MPSolver::CreateSolver("SCIP"));

    if (!solver) {
      fmt::print("Solver not available\n");
    }

    infinity = solver->infinity();
  }

  std::vector<MPVariable *> initiallyAllocatedOnDevice;
  std::vector<std::vector<MPVariable *>> p;
  std::vector<std::vector<std::vector<MPVariable *>>> o;

  void defineDecisionVariables(const SecondStepInput &secondStepInput) {
    // Add decision variables for initially allocated on device or not
    initiallyAllocatedOnDevice.clear();
    for (int i = 0; i < numberOfArrays; i++) {
      initiallyAllocatedOnDevice.push_back(solver->MakeBoolVar(fmt::format("I_{{{}}}", i)));
      initiallyAllocatedOnDevice[i]->SetUB(0);
    }
    for (auto arr : secondStepInput.applicationInputArrays) {
      initiallyAllocatedOnDevice[arr]->SetUB(1);
    }

    // Add decision variables for prefetching
    p.clear();
    for (int i = 0; i < numberOfKernels; i++) {
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
    for (int i = 0; i < numberOfKernels; i++) {
      o.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        o[i].push_back({});

        auto sumLessThanOneConstraint = solver->MakeRowConstraint(0, 1);

        sumLessThanOneConstraint->SetCoefficient(p[i][j], 1);

        for (int k = 0; k < numberOfKernels; k++) {
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

  std::vector<std::vector<MPVariable *>> x;
  std::vector<std::vector<MPVariable *>> y;

  void defineXAndY(const SecondStepInput &secondStepInput) {
    // Define the variables representing whether the memory for an array is allocated on the device
    x.clear();
    for (int i = 0; i < numberOfKernels; i++) {
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
    for (int i = 0; i < numberOfKernels; i++) {
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
          for (int v = u + 1; v < numberOfKernels; v++) {
            constraint->SetCoefficient(o[u][j][v], -1);
          }
        }

        auto offloadingConstraint = solver->MakeRowConstraint(0, 1);
        offloadingConstraint->SetCoefficient(y[i][j], 1);
        for (int k = i + 1; k < numberOfKernels; k++) {
          offloadingConstraint->SetCoefficient(o[i][j][k], -1);
        }
      }
    }
  }

  std::vector<MPVariable *> w;
  std::vector<double> arrayPrefetchingTimes, arrayOffloadingTimes;

  void defineWeights(const SecondStepInput &secondStepInput) {
    w.clear();
    w.resize(numberOfVertices);

    // Add weights for kernel vertices
    for (int i = 0; i < numberOfKernels; i++) {
      w[getKernelStartVertexIndex(i)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getKernelStartVertexIndex(i)));
      w[getKernelVertexIndex(i)] = solver->MakeNumVar(
        secondStepInput.kernelRunningTimes[i],
        secondStepInput.kernelRunningTimes[i],
        fmt::format("w_{{{}}}", getKernelVertexIndex(i))
      );
    }

    // Preprocess weights for data movement vertices
    arrayPrefetchingTimes.clear();
    arrayOffloadingTimes.clear();
    for (int i = 0; i < numberOfArrays; i++) {
      arrayPrefetchingTimes.push_back(secondStepInput.arraySizes[i] / secondStepInput.prefetchingBandwidth);
      arrayOffloadingTimes.push_back(secondStepInput.arraySizes[i] / secondStepInput.offloadingBandwidth);
    }

    // Add weights for prefetches
    for (int i = 0; i < numberOfKernels; i++) {
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
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldDeallocate[std::make_pair(i, j)]) {
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(0, 0, fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j)));
        } else {
          w[getOffloadVertexIndex(i, j)] = solver->MakeNumVar(0, arrayOffloadingTimes[j], fmt::format("w_{{{}}}", getOffloadVertexIndex(i, j)));

          auto constraint = solver->MakeRowConstraint(0, 0);
          constraint->SetCoefficient(w[getOffloadVertexIndex(i, j)], -1);
          for (int k = 0; k < numberOfKernels; k++) {
            constraint->SetCoefficient(o[i][j][k], arrayOffloadingTimes[j]);
          }
        }
      }
    }
  }

  std::vector<std::vector<MPVariable *>> e;

  void addEdges(const SecondStepInput &secondStepInput) {
    auto zeroConstant = solver->MakeIntVar(0, 0, "zero");
    auto oneConstant = solver->MakeIntVar(1, 1, "one");

    e.clear();
    e.resize(numberOfVertices, std::vector<MPVariable *>(numberOfVertices, zeroConstant));

    // Add edges between kernel and kernel start vertices
    for (int i = 0; i < numberOfKernels; i++) {
      e[getKernelStartVertexIndex(i)][getKernelVertexIndex(i)] = oneConstant;
      if (i > 0) {
        e[getKernelVertexIndex(i - 1)][getKernelStartVertexIndex(i)] = oneConstant;
      }
    }

    // Add edges for prefetching vertices
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getKernelStartVertexIndex(i)][getPrefetchVertexIndex(i, j)] = p[i][j];
        for (int k = i; k < numberOfKernels; k++) {
          if (secondStepInput.kernelInputArrays[k].count(j) != 0 || secondStepInput.kernelOutputArrays[k].count(j) != 0) {
            e[getPrefetchVertexIndex(i, j)][getKernelVertexIndex(k)] = oneConstant;
          }
        }
      }
    }

    // Serialize prefetches
    for (int i = 0; i < numberOfKernels; i++) {
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
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getKernelVertexIndex(i)][getOffloadVertexIndex(i, j)] = solver->MakeBoolVar("");

        auto constraint = solver->MakeRowConstraint(0, 0);
        constraint->SetCoefficient(e[getKernelVertexIndex(i)][getOffloadVertexIndex(i, j)], -1);
        for (int k = 0; k < numberOfKernels; k++) {
          constraint->SetCoefficient(o[i][j][k], 1);
        }

        for (int k = i + 1; k < numberOfKernels; k++) {
          e[getOffloadVertexIndex(i, j)][getKernelStartVertexIndex(k)] = o[i][j][k];
        }
      }
    }

    // Serialize offloadings
    for (int i = 0; i < numberOfKernels; i++) {
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

  std::vector<MPVariable *> z;

  void addLongestPathConstraints(const SecondStepInput &secondStepInput) {
    z.clear();

    for (int u = 0; u < numberOfVertices; u++) {
      if (u == getKernelStartVertexIndex(0)) {
        z.push_back(solver->MakeNumVar(0, 0, ""));
      } else {
        z.push_back(solver->MakeNumVar(0, infinity, ""));
      }
    }

    for (int u = 0; u < numberOfVertices; u++) {
      if (u != getKernelStartVertexIndex(0)) {
        for (int v = 0; v < numberOfVertices; v++) {
          auto oneMinusE = solver->MakeBoolVar("");
          auto oneMinusEConstraint = solver->MakeRowConstraint(1, 1);
          oneMinusEConstraint->SetCoefficient(oneMinusE, 1);
          oneMinusEConstraint->SetCoefficient(e[v][u], 1);

          auto constraint = solver->MakeRowConstraint(0, infinity);
          constraint->SetCoefficient(z[u], 1);
          constraint->SetCoefficient(z[v], -1);
          constraint->SetCoefficient(w[u], -1);
          constraint->SetCoefficient(oneMinusE, infinity);
        }
      }
    }

    auto zLastKernelConstraint = solver->MakeRowConstraint(0, NUMBER_OF_KERNELS * KERNEL_RUNNING_TIME * ACCEPTABLE_RUNNING_TIME_FACTOR);
    zLastKernelConstraint->SetCoefficient(z[getKernelVertexIndex(numberOfKernels - 1)], 1);
  }

  void addKernelDataDependencyConstraints(const SecondStepInput &secondStepInput) {
    for (int i = 0; i < numberOfKernels; i++) {
      for (auto &arr : secondStepInput.kernelInputArrays[i]) {
        auto constraint = solver->MakeRowConstraint(1, 1);
        constraint->SetCoefficient(y[i][arr], 1);
      }
      for (auto &arr : secondStepInput.kernelOutputArrays[i]) {
        auto constraint = solver->MakeRowConstraint(1, 1);
        constraint->SetCoefficient(y[i][arr], 1);
      }
    }
  }

  void solveSecondStep(const SecondStepInput &secondStepInput) {
    this->initialize();

    this->preprocessSecondStepInput(secondStepInput);

    this->defineDecisionVariables(secondStepInput);

    this->defineXAndY(secondStepInput);

    this->defineWeights(secondStepInput);

    this->addEdges(secondStepInput);

    this->addLongestPathConstraints(secondStepInput);

    this->addKernelDataDependencyConstraints(secondStepInput);

    // Represent the peak memory usage
    auto peakMemoryUsage = solver->MakeNumVar(0, infinity, "");
    for (int i = 0; i < numberOfKernels; i++) {
      auto constraint = solver->MakeRowConstraint(0, infinity);
      constraint->SetCoefficient(peakMemoryUsage, 1);
      for (int j = 0; j < numberOfArrays; j++) {
        constraint->SetCoefficient(x[i][j], -static_cast<double>(secondStepInput.arraySizes[j]));
      }
    }

    // Objectives:
    // 1. Minimize the peak memory usage
    // 2. Minimize the total running time
    // Objectives are in Lexicographic Combination.
    // Z3 solves the first for the objective that is declared first.
    auto obj1 = solver->MutableObjective();
    obj1->SetCoefficient(peakMemoryUsage, 1);
    obj1->SetMinimization();

    auto resultStatus = solver->Solve();

    if (resultStatus == MPSolver::OPTIMAL) {
      auto optimizedPeakMemoryUsage = obj1->Value();
      auto totalRunningTime = z[getKernelVertexIndex(numberOfKernels - 1)]->solution_value();

      fmt::print("Optimal peak memory usage (Byte): {:.2f}\n", optimizedPeakMemoryUsage);
      fmt::print("Optimized peak memory usage / original: {:.6f}%\n", optimizedPeakMemoryUsage / (NUMBER_OF_KERNELS * ARRAY_SIZE * 2.0 + ARRAY_SIZE) * 100.0);

      fmt::print("Total running time (s): {:.6f}\n", totalRunningTime);
      fmt::print("Total running time / original: {:.6f}%\n", totalRunningTime / (NUMBER_OF_KERNELS * KERNEL_RUNNING_TIME) * 100.0);

      fmt::print("---\nSolution:\n");

      for (int i = 0; i < numberOfKernels; i++) {
        fmt::print("K_{{{}}}\n", i);
        for (int j = 0; j < 3; j++) {
          const int arrayIndex = i * 3 + j;
          fmt::print(
            "  I_{{{}}} = {} {}\n",
            arrayIndex,
            initiallyAllocatedOnDevice[arrayIndex]->solution_value(),
            arrayIndex % 3 == 2 ? "(Output)" : ""
          );
        }
      }

      fmt::print("\n");

      for (int i = 0; i < numberOfKernels; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          if (p[i][j]->solution_value() > 0) {
            fmt::print("p_{{{}, {}}} = {}\n", i, j, true);
          }
        }
      }

      fmt::print("\n");

      for (int i = 0; i < numberOfKernels; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          for (int k = 0; k < numberOfKernels; k++) {
            if (o[i][j][k]->solution_value() > 0) {
              fmt::print("o_{{{}, {}, {}}} = {}\n", i, j, k, true);
            }
          }
        }
      }
    } else {
      fmt::print("No solution found.\n");
    }
  }

  void runSecondStepExample() {
    auto exampleSecondStepInput = TwoStepIntegerProgrammingStrategy::getExampleSecondStepInput();
    this->solveSecondStep(exampleSecondStepInput);
  }
};

int main() {
  // operations_research::BasicExample();

  auto twoStepIntegerProgrammingStrategy = std::make_unique<TwoStepIntegerProgrammingStrategy>();
  twoStepIntegerProgrammingStrategy->runSecondStepExample();

  return 0;
}