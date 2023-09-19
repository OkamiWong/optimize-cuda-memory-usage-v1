#include <fmt/core.h>
#include <z3++.h>

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

void iteExample2() {
  std::cout << "if-then-else example2\n";
  z3::context c;
  z3::expr b = c.bool_const("b");
  z3::expr x = c.int_const("x");
  z3::expr y = c.int_const("y");
  std::cout << (z3::ite(b, x, y) > 0) << "\n";
}

void longestPathExample() {
  constexpr int S = 0;
  constexpr int T = 2;
  // Adjacency matrix
  std::vector<std::vector<double>> dis = {
    {0, 1, 3, 0},
    {0, 0, 3, 0},
    {0, 0, 0, 0},
    {0, 0, 1, 0},
  };

  const int n = dis.size();

  z3::context context;
  z3::optimize optimize(context);

  std::vector<z3::expr> z;

  for (int i = 0; i < n; i++) {
    if (i == S) {
      z.push_back(context.real_val(0));
    } else {
      z.push_back(context.real_const(fmt::format("z[{}]", i).c_str()));
    }
  }

  auto minusInfinity = context.real_val(-(0x7fffffff));

  for (int i = 1; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(dis[j][i]) <= std::numeric_limits<double>::epsilon()) {
        optimize.add(z[i] >= z[j] + minusInfinity);
      } else {
        optimize.add(z[i] >= z[j] + context.real_val(std::to_string(dis[j][i]).c_str()));
      }
    }
  }

  std::cout << optimize << std::endl;

  auto handle = optimize.minimize(z[T]);
  if (optimize.check() == z3::check_result::sat) {
    std::cout << optimize.lower(handle) << std::endl;
  } else {
    std::cout << "No solution found" << std::endl;
  }
}

struct TwoStepIntegerProgrammingStrategy {
  // Experiment setup
  static constexpr int NUMBER_OF_KERNELS = 4;
  static constexpr double KERNEL_RUNNING_TIME = 1;
  static constexpr double CONNECTION_BANDWIDTH = 281.0 * 1e9;
  static constexpr size_t ARRAY_SIZE = CONNECTION_BANDWIDTH * KERNEL_RUNNING_TIME / 2.0 * 2.0;

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

  std::unique_ptr<z3::context> context;
  std::unique_ptr<z3::optimize> optimize;

  void initializeZ3() {
    context = std::make_unique<z3::context>();
    optimize = std::make_unique<z3::optimize>(*context);
  }

  std::vector<z3::expr> initiallyAllocatedOnDevice;
  std::vector<std::vector<z3::expr>> p;
  std::vector<std::vector<std::vector<z3::expr>>> o;

  void defineDecisionVariables(const SecondStepInput &secondStepInput) {
    // Add decision variables for initially allocated on device or not
    initiallyAllocatedOnDevice.clear();
    initiallyAllocatedOnDevice.resize(numberOfArrays, context->bool_val(false));
    for (auto arr : secondStepInput.applicationInputArrays) {
      initiallyAllocatedOnDevice[arr] = context->bool_const(fmt::format("I_{{{}}}", arr).c_str());
    }

    // Add decision variables for prefetching
    p.clear();
    for (int i = 0; i < numberOfKernels; i++) {
      p.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldAllocate[std::make_pair(i, j)]) {
          p[i].push_back(context->bool_val(true));
        } else {
          p[i].push_back(context->bool_const(fmt::format("p_{{{},{}}}", i, j).c_str()));
        }
      }
    }

    // Add decision variables for offloading
    o.clear();
    for (int i = 0; i < numberOfKernels; i++) {
      o.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        o[i].push_back({});
        if (shouldDeallocate[std::make_pair(i, j)]) {
          for (int k = 0; k < numberOfKernels; k++) {
            if (k == i + 1) {
              o[i][j].push_back(context->bool_val(true));
            } else {
              o[i][j].push_back(context->bool_val(false));
            }
          }
        } else {
          for (int k = 0; k < numberOfKernels; k++) {
            if (k <= i) {
              o[i][j].push_back(context->bool_val(false));
            } else {
              o[i][j].push_back(context->bool_const(fmt::format("o_{{{},{},{}}}", i, j, k).c_str()));
              optimize->add(!(o[i][j][k] && p[i][j]));
            }
          }
        }
      }
    }
  }

  std::vector<std::vector<z3::expr>> x;
  std::vector<std::vector<z3::expr>> y;

  void defineXAndY(const SecondStepInput &secondStepInput) {
    // Define the variables representing whether the memory for an array is allocated on the device
    x.clear();
    for (int i = 0; i < numberOfKernels; i++) {
      x.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        auto rhs = z3::ite(initiallyAllocatedOnDevice[j], context->int_val(1), context->int_val(0));

        for (int u = 0; u <= i; u++) {
          rhs = rhs + z3::ite(p[u][j], context->int_val(1), context->int_val(0));
        }

        for (int u = 0; u <= i - 1; u++) {
          for (int v = u + 1; v <= i; v++) {
            rhs = rhs - z3::ite(o[u][j][v], context->int_val(1), context->int_val(0));
          }
        }
        x[i].push_back(rhs);
        optimize->add(x[i][j] >= 0);
        optimize->add(x[i][j] <= 1);
      }
    }

    // Define the variables representing whether an array is available
    y.clear();
    for (int i = 0; i < numberOfKernels; i++) {
      y.push_back({});
      for (int j = 0; j < numberOfArrays; j++) {
        auto rhs = z3::ite(initiallyAllocatedOnDevice[j], context->int_val(1), context->int_val(0));

        for (int u = 0; u <= i; u++) {
          rhs = rhs + z3::ite(p[u][j], context->int_val(1), context->int_val(0));
        }

        for (int u = 0; u <= i - 1; u++) {
          for (int v = u + 1; v < numberOfKernels; v++) {
            rhs = rhs - z3::ite(o[u][j][v], context->int_val(1), context->int_val(0));
          }
        }

        y[i].push_back(rhs);
        optimize->add(y[i][j] >= 0);
        optimize->add(y[i][j] <= 1);
      }
    }
  }

  std::vector<z3::expr> w;
  std::vector<z3::expr> arrayPrefetchingTimes, arrayOffloadingTimes;

  void defineWeights(const SecondStepInput &secondStepInput) {
    w.clear();
    w.resize(numberOfVertices, context->real_val(0));

    // Add weights for kernel vertices
    for (int i = 0; i < numberOfKernels; i++) {
      w[getKernelStartVertexIndex(i)] = context->real_val(0);
      w[getKernelVertexIndex(i)] = context->real_val(fmt::format("{:.6f}", secondStepInput.kernelRunningTimes[i]).c_str());
    }

    // Preprocess weights for data movement vertices
    arrayPrefetchingTimes.clear();
    arrayOffloadingTimes.clear();
    for (int i = 0; i < numberOfArrays; i++) {
      double prefetchingTime = secondStepInput.arraySizes[i] / secondStepInput.prefetchingBandwidth;
      double offloadingTime = secondStepInput.arraySizes[i] / secondStepInput.offloadingBandwidth;
      arrayPrefetchingTimes.push_back(context->real_val(fmt::format("{:.6f}", prefetchingTime).c_str()));
      arrayOffloadingTimes.push_back(context->real_val(fmt::format("{:.6f}", offloadingTime).c_str()));
    }

    // Add weights for prefetches
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldAllocate[std::make_pair(i, j)]) {
          w[getPrefetchVertexIndex(i, j)] = context->real_val(0);
        } else {
          w[getPrefetchVertexIndex(i, j)] = z3::ite(p[i][j], arrayPrefetchingTimes[j], context->real_val(0));
        }
      }
    }

    // Add weights for offloadings
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (shouldDeallocate[std::make_pair(i, j)]) {
          w[getOffloadVertexIndex(i, j)] = context->real_val(0);
        } else {
          auto rhs = context->int_val(0);
          for (int k = 0; k < numberOfKernels; k++) {
            rhs = rhs + z3::ite(o[i][j][k], context->int_val(1), context->int_val(0));
          }
          optimize->add(rhs <= 1);

          w[getOffloadVertexIndex(i, j)] = rhs * arrayOffloadingTimes[j];
        }
      }
    }
  }

  std::vector<std::vector<z3::expr>> e;

  void addEdges(const SecondStepInput &secondStepInput) {
    e.clear();
    e.resize(numberOfVertices, std::vector<z3::expr>(numberOfVertices, context->bool_val(false)));

    // Add edges between kernel and kernel start vertices
    for (int i = 0; i < numberOfKernels; i++) {
      e[getKernelStartVertexIndex(i)][getKernelVertexIndex(i)] = context->bool_val(true);
      if (i > 0) {
        e[getKernelVertexIndex(i - 1)][getKernelStartVertexIndex(i)] = context->bool_val(true);
      }
    }

    // Add edges for prefetching vertices
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        e[getKernelStartVertexIndex(i)][getPrefetchVertexIndex(i, j)] = p[i][j];
        for (int k = i; k < numberOfKernels; k++) {
          if (secondStepInput.kernelInputArrays[k].count(j) != 0 || secondStepInput.kernelOutputArrays[k].count(j) != 0) {
            e[getPrefetchVertexIndex(i, j)][getKernelVertexIndex(k)] = context->bool_val(true);
            break;
          }
        }
      }
    }

    // Serialize prefetches
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        if (j == 0) {
          if (i != 0) {
            e[getPrefetchVertexIndex(i - 1, numberOfArrays - 1)][getPrefetchVertexIndex(i, j)] = context->bool_val(true);
          }
        } else {
          e[getPrefetchVertexIndex(i, j - 1)][getPrefetchVertexIndex(i, j)] = context->bool_val(true);
        }
      }
    }

    // Add edges for offloading vertices
    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        auto rhs = o[i][j][0];
        for (int k = 1; k < numberOfKernels; k++) {
          rhs = rhs || o[i][j][k];
        }

        e[getKernelVertexIndex(i)][getOffloadVertexIndex(i, j)] = rhs;
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
            e[getOffloadVertexIndex(i - 1, numberOfArrays - 1)][getOffloadVertexIndex(i, j)] = context->bool_val(true);
          }
        } else {
          e[getOffloadVertexIndex(i, j - 1)][getOffloadVertexIndex(i, j)] = context->bool_val(true);
        }
      }
    }
  }

  std::vector<z3::expr> z;

  void addLongestPathConstraints(const SecondStepInput &secondStepInput) {
    z.clear();
    z.resize(numberOfVertices, context->real_val(0));

    z[getKernelStartVertexIndex(0)] = context->real_val(0);
    for (int u = 1; u < numberOfVertices; u++) {
      z[u] = context->real_const(fmt::format("z_{{{}}}", u).c_str());
    }

    for (int u = 1; u < numberOfVertices; u++) {
      for (int v = 0; v < numberOfVertices; v++) {
        optimize->add(z[u] >= z[v] + w[u] + z3::ite(e[v][u], context->real_val(0), context->real_val(std::numeric_limits<int>::min())));
      }
    }

    optimize->add(z[getKernelVertexIndex(numberOfKernels - 1)] <= context->real_val(fmt::format("{:.6f}", NUMBER_OF_KERNELS * KERNEL_RUNNING_TIME * 1.1).c_str()));
  }

  void addKernelDataDependencyConstraints(const SecondStepInput &secondStepInput) {
    for (int i = 0; i < numberOfKernels; i++) {
      for (auto &arr : secondStepInput.kernelInputArrays[i]) {
        optimize->add(y[i][arr] == 1);
      }
      for (auto &arr : secondStepInput.kernelOutputArrays[i]) {
        optimize->add(y[i][arr] == 1);
      }
    }
  }

  void solveSecondStep(const SecondStepInput &secondStepInput) {
    this->initializeZ3();

    this->preprocessSecondStepInput(secondStepInput);

    this->defineDecisionVariables(secondStepInput);

    this->defineXAndY(secondStepInput);

    this->defineWeights(secondStepInput);

    this->addEdges(secondStepInput);

    this->addLongestPathConstraints(secondStepInput);

    this->addKernelDataDependencyConstraints(secondStepInput);

    // Represent the peak memory usage
    auto peakMemoryUsage = context->real_const("peakMemoryUsage");
    for (int i = 0; i < numberOfKernels; i++) {
      auto rhs = context->real_val(secondStepInput.arraySizes[i]) * x[i][0];
      for (int j = 1; j < numberOfArrays; j++) {
        rhs = rhs + context->real_val(secondStepInput.arraySizes[j]) * x[i][j];
      }
      optimize->add(peakMemoryUsage >= rhs);
    }

    // Objectives:
    // 1. Minimize the peak memory usage
    // 2. Minimize the total running time
    // Objectives are in Lexicographic Combination.
    // Z3 solves the first for the objective that is declared first.
    auto objective1 = optimize->minimize(peakMemoryUsage);
    auto objective2 = optimize->minimize(z[getKernelVertexIndex(numberOfKernels - 1)]);

    // Solve and print result
    if (optimize->check() == z3::check_result::sat) {
      auto model = optimize->get_model();
      auto optimizedPeakMemoryUsage = optimize->lower(objective1).get_numeral_int64();
      auto totalRunningTime = optimize->lower(objective2).as_double();

      fmt::print("Optimal peak memory usage (Byte): {}\n", optimizedPeakMemoryUsage);
      fmt::print("Optimized peak memory usage / original: {:.6f}%\n", static_cast<double>(optimizedPeakMemoryUsage) / (NUMBER_OF_KERNELS * ARRAY_SIZE * 3.0) * 100.0);

      fmt::print("Total running time (s): {:.6f}\n", totalRunningTime);
      fmt::print("Total running time / original: {:.6f}%\n", totalRunningTime / (NUMBER_OF_KERNELS * KERNEL_RUNNING_TIME) * 100.0);

      fmt::print("---\nSolution:\n");

      for (int i = 0; i < numberOfArrays; i++) {
        fmt::print("I_{{{}}} = {}\n", i, model.eval(initiallyAllocatedOnDevice[i]).is_true());
      }

      fmt::print("\n");

      for (int i = 0; i < numberOfKernels; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          if (model.eval(p[i][j]).is_true()) {
            fmt::print("p_{{{}, {}}} = {}\n", i, j, true);
          }
        }
      }

      fmt::print("\n");

      for (int i = 0; i < numberOfKernels; i++) {
        for (int j = 0; j < numberOfArrays; j++) {
          for (int k = 0; k < numberOfKernels; k++) {
            if (model.eval(o[i][j][k]).is_true()) {
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
  // iteExample2();
  // longestPathExample();

  auto twoStepIntegerProgrammingStrategy = std::make_unique<TwoStepIntegerProgrammingStrategy>();
  twoStepIntegerProgrammingStrategy->runSecondStepExample();

  return 0;
}
