#include <fmt/core.h>
#include <z3++.h>

#include <cmath>
#include <iostream>
#include <limits>
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

struct OptimizationInput {
  std::vector<int> kernelExecutionSequence;
  std::vector<double> kernelRunningTimes;

  std::vector<size_t> arraySizes;
  std::vector<bool> arrayInitiallyOnDevice;
  std::vector<std::vector<int>> kernelDataDependencies;

  double prefetchingBandwidth, offloadingBandwidth;
};

constexpr int NUMBER_OF_KERNELS = 4;
constexpr int EXPECTED_PREFETCH_START_KERNEL = 1;
constexpr int EXPECTED_PREFETCH_CYCLE = 2;
constexpr double KERNEL_RUNNING_TIME = 1;
constexpr double CONNECTION_BANDWIDTH = 281.0 * 1e9;
constexpr size_t ARRAY_SIZE = CONNECTION_BANDWIDTH * KERNEL_RUNNING_TIME / 3.0 * 2.0;

OptimizationInput getChainOfStreamKernelsExampleOptimizationInput() {
  OptimizationInput input;

  for (int i = 0; i < NUMBER_OF_KERNELS; i++) {
    input.kernelExecutionSequence.push_back(i);
    input.kernelRunningTimes.push_back(KERNEL_RUNNING_TIME);
    input.kernelDataDependencies.push_back({});
    for (int j = i * 3; j < (i + 1) * 3; j++) {
      input.kernelDataDependencies[i].push_back(j);
    }
  }

  for (int i = 0; i < NUMBER_OF_KERNELS; i++) {
    for (int j = i * 3; j < (i + 1) * 3; j++) {
      input.arrayInitiallyOnDevice.push_back(true);
      input.arraySizes.push_back(ARRAY_SIZE);
      if (i % EXPECTED_PREFETCH_CYCLE == EXPECTED_PREFETCH_START_KERNEL && i != EXPECTED_PREFETCH_START_KERNEL) {
        input.arrayInitiallyOnDevice[j] = false;
      }
    }
  }

  input.prefetchingBandwidth = input.offloadingBandwidth = CONNECTION_BANDWIDTH;

  return input;
}

void chainOfStreamKernelsExample() {
  auto input = getChainOfStreamKernelsExampleOptimizationInput();

  const int numberOfKernels = input.kernelExecutionSequence.size();
  const int numberOfArrays = input.arraySizes.size();
  const int numberOfVertices =
    numberOfKernels * 2 + numberOfKernels * numberOfArrays + numberOfKernels * numberOfArrays;

  const auto getKernelVertexIndex = [=](int i) {
    return i * 2 + 1;
  };

  const auto getKernelStartVertexIndex = [=](int i) {
    return i * 2;
  };

  const auto getPrefetchVertexIndex = [=](int i, int j) {
    return numberOfKernels * 2 + i * numberOfArrays + j;
  };

  const auto getOffloadVertexIndex = [=](int i, int j) {
    return numberOfKernels * 2 + numberOfKernels * numberOfArrays + i * numberOfArrays + j;
  };

  z3::context context;
  z3::optimize optimize(context);

  const auto trueBoolConstantExpr = context.bool_val(true);
  const auto falseBoolConstantExpr = context.bool_val(false);
  const auto oneIntConstantExpr = context.int_val(1);
  const auto zeroIntConstantExpr = context.int_val(0);
  const auto zeroRealConstantExpr = context.real_val(0);
  const auto minusInfinityRealConstantExpr = context.real_val(-(0x7fffffff));

  // Add decision variables for prefetching
  std::vector<std::vector<z3::expr>> p;
  for (int i = 0; i < numberOfKernels; i++) {
    p.push_back({});
    for (int j = 0; j < numberOfArrays; j++) {
      p[i].push_back(context.bool_const(fmt::format("p_{{{},{}}}", i, j).c_str()));
    }
  }

  // Add decision variables for offloading
  std::vector<std::vector<std::vector<z3::expr>>> o;
  for (int i = 0; i < numberOfKernels; i++) {
    o.push_back({});
    for (int j = 0; j < numberOfArrays; j++) {
      o[i].push_back({});
      for (int k = 0; k < numberOfKernels; k++) {
        if (k <= i) {
          o[i][j].push_back(falseBoolConstantExpr);
        } else {
          o[i][j].push_back(context.bool_const(fmt::format("o_{{{},{},{}}}", i, j, k).c_str()));
          optimize.add(!(o[i][j][k] && p[i][j]));
        }
      }
    }
  }

  // Define the variables representing whether the memory for an array is allocated on the device
  std::vector<std::vector<z3::expr>> x;
  for (int i = 0; i < numberOfKernels; i++) {
    x.push_back({});
    for (int j = 0; j < numberOfArrays; j++) {
      auto rhs = context.int_val(input.arrayInitiallyOnDevice[j]);
      for (int u = 0; u <= i; u++) {
        rhs = rhs + z3::ite(p[u][j], oneIntConstantExpr, zeroIntConstantExpr);
        for (int v = u + 1; v <= i; v++) {
          rhs = rhs - z3::ite(o[u][j][v], oneIntConstantExpr, zeroIntConstantExpr);
        }
      }
      x[i].push_back(rhs);
      optimize.add(x[i][j] >= 0);
      optimize.add(x[i][j] <= 1);
    }
  }

  // Define the variables representing whether an array is available
  std::vector<std::vector<z3::expr>> y;
  for (int i = 0; i < numberOfKernels; i++) {
    y.push_back({});
    for (int j = 0; j < numberOfArrays; j++) {
      auto rhs = context.int_val(input.arrayInitiallyOnDevice[j]);
      for (int u = 0; u <= i; u++) {
        rhs = rhs + z3::ite(p[u][j], oneIntConstantExpr, zeroIntConstantExpr);
        for (int v = u + 1; v < numberOfKernels; v++) {
          rhs = rhs - z3::ite(o[u][j][v], oneIntConstantExpr, zeroIntConstantExpr);
        }
      }
      y[i].push_back(rhs);
      optimize.add(y[i][j] >= 0);
      optimize.add(y[i][j] <= 1);
    }
  }

  // Add weights for kernel vertices
  std::vector<z3::expr> w(numberOfVertices, zeroRealConstantExpr);
  for (int i = 0; i < numberOfKernels; i++) {
    w[getKernelStartVertexIndex(i)] = zeroRealConstantExpr;
    w[getKernelVertexIndex(i)] = context.real_val(fmt::format("{:.6f}", input.kernelRunningTimes[i]).c_str());
  }

  // Preprocess weights for data movement vertices
  std::vector<z3::expr> arrayPrefetchingTimes, arrayOffloadingTimes;
  for (int i = 0; i < numberOfArrays; i++) {
    double prefetchingTime = input.arraySizes[i] / input.prefetchingBandwidth;
    double offloadingTime = input.arraySizes[i] / input.offloadingBandwidth;
    arrayPrefetchingTimes.push_back(context.real_val(fmt::format("{:.6f}", prefetchingTime).c_str()));
    arrayOffloadingTimes.push_back(context.real_val(fmt::format("{:.6f}", offloadingTime).c_str()));
  }

  // Add weights for data movements
  for (int i = 0; i < numberOfKernels; i++) {
    for (int j = 0; j < numberOfArrays; j++) {
      w[getPrefetchVertexIndex(i, j)] = z3::ite(p[i][j], arrayPrefetchingTimes[j], zeroRealConstantExpr);

      auto rhs = zeroIntConstantExpr;
      for (int k = 0; k < numberOfKernels; k++) {
        rhs = rhs + z3::ite(o[i][j][k], oneIntConstantExpr, zeroIntConstantExpr);
      }
      optimize.add(rhs <= 1);

      w[getOffloadVertexIndex(i, j)] = rhs * arrayOffloadingTimes[j];
    }
  }

  // Add edges
  std::vector<std::vector<z3::expr>> e(numberOfVertices, std::vector<z3::expr>(numberOfVertices, falseBoolConstantExpr));

  // Add edges between kernel and kernel start vertices
  for (int i = 0; i < numberOfKernels; i++) {
    e[getKernelStartVertexIndex(i)][getKernelVertexIndex(i)] = trueBoolConstantExpr;
    if (i > 0) {
      e[getKernelVertexIndex(i - 1)][getKernelStartVertexIndex(i)] = trueBoolConstantExpr;
    }
  }

  // Add edges for prefetching vertices
  for (int i = 0; i < numberOfKernels; i++) {
    for (int j = 0; j < numberOfArrays; j++) {
      e[getKernelStartVertexIndex(i)][getPrefetchVertexIndex(i, j)] = p[i][j];
      for (int k = i; k < numberOfKernels; k++) {
        for (auto &arr : input.kernelDataDependencies[k]) {
          if (arr == j) {
            e[getPrefetchVertexIndex(i, j)][getKernelVertexIndex(k)] = trueBoolConstantExpr;
            break;
          }
        }
      }
    }
  }

  // Serialize prefetches
  for (int i = 0; i < numberOfKernels; i++) {
    for (int j = 0; j < numberOfArrays; j++) {
      if (j == 0) {
        if (i != 0) {
          e[getPrefetchVertexIndex(i - 1, numberOfArrays - 1)][getPrefetchVertexIndex(i, j)] = trueBoolConstantExpr;
        }
      } else {
        e[getPrefetchVertexIndex(i, j - 1)][getPrefetchVertexIndex(i, j)] = trueBoolConstantExpr;
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

      e[getKernelStartVertexIndex(i)][getOffloadVertexIndex(i, j)] = rhs;
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
          e[getOffloadVertexIndex(i - 1, numberOfArrays - 1)][getOffloadVertexIndex(i, j)] = trueBoolConstantExpr;
        }
      } else {
        e[getOffloadVertexIndex(i, j - 1)][getOffloadVertexIndex(i, j)] = trueBoolConstantExpr;
      }
    }
  }

  // Constraints for the longest path
  std::vector<z3::expr> z(numberOfVertices, zeroRealConstantExpr);

  z[getKernelStartVertexIndex(0)] = zeroRealConstantExpr;
  for (int u = 1; u < numberOfVertices; u++) {
    z[u] = context.real_const(fmt::format("z_{{{}}}", u).c_str());
  }

  for (int u = 1; u < numberOfVertices; u++) {
    for (int v = 0; v < numberOfVertices; v++) {
      optimize.add(z[u] >= z[v] + w[u] + z3::ite(e[v][u], zeroRealConstantExpr, minusInfinityRealConstantExpr));
    }
  }

  optimize.add(z[getKernelVertexIndex(numberOfKernels - 1)] <= context.real_val(fmt::format("{:.6f}", NUMBER_OF_KERNELS * KERNEL_RUNNING_TIME * 1.1).c_str()));

  // Constraints for meeting the data dependencies of each kernel
  for (int i = 0; i < numberOfKernels; i++) {
    for (auto &arr : input.kernelDataDependencies[i]) {
      optimize.add(y[i][arr] == 1);
    }
  }

  // Represent the peak memory usage
  auto peakMemoryUsage = context.real_const("peakMemoryUsage");
  for (int i = 0; i < numberOfKernels; i++) {
    auto rhs = context.real_val(input.arraySizes[i]) * x[i][0];
    for (int j = 1; j < numberOfArrays; j++) {
      rhs = rhs + context.real_val(input.arraySizes[j]) * x[i][j];
    }
    optimize.add(peakMemoryUsage >= rhs);
  }

  // Objective
  auto objective1 = optimize.minimize(peakMemoryUsage);
  auto objective2 = optimize.minimize(z[getKernelVertexIndex(numberOfKernels - 1)]);

  // Solve
  if (optimize.check() == z3::check_result::sat) {
    auto model = optimize.get_model();
    auto optimizedPeakMemoryUsage = optimize.lower(objective1).get_numeral_int64();
    auto totalRunningTime = optimize.lower(objective2).as_double();

    fmt::print("Optimal peak memory usage (Byte): {}\n", optimizedPeakMemoryUsage);
    fmt::print("Optimized peak memory usage / original: {:.6f}%\n", static_cast<double>(optimizedPeakMemoryUsage) / (NUMBER_OF_KERNELS * ARRAY_SIZE * 3.0) * 100.0);

    fmt::print("Total running time (s): {:.6f}\n", totalRunningTime);
    fmt::print("Total running time / original: {:.6f}%\n", totalRunningTime / (NUMBER_OF_KERNELS * KERNEL_RUNNING_TIME) * 100.0);

    fmt::print("---\nInitial data distribution:\n");
    for (int i = 0; i < numberOfArrays; i++) {
      fmt::print("I_{{{}}} = {}\n", i, static_cast<int>(input.arrayInitiallyOnDevice[i]));
    }

    fmt::print("---\nSolution:\n");

    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        fmt::print("p_{{{}, {}}} = {}, ", i, j, model.eval(p[i][j]).is_true());
        fmt::print("\n");
      }
    }

    for (int i = 0; i < numberOfKernels; i++) {
      for (int j = 0; j < numberOfArrays; j++) {
        for (int k = 0; k < numberOfKernels; k++) {
          fmt::print("o_{{{}, {}, {}}} = {}, ", i, j, k, model.eval(o[i][j][k]).is_true());
        }
        fmt::print("\n");
      }
    }

    // for (int i = 0; i < numberOfKernels; i++) {
    //   fmt::print("z[{} Start] = {}; z[{}] = {}\n", i, model.eval(z[getKernelStartVertexIndex(i)]).get_decimal_string(6), i, model.eval(z[getKernelVertexIndex(i)]).get_decimal_string(6));
    // }

    // auto printZPrefetch = [&](int i, int j) {
    //   fmt::print("z[P_{{{}, {}}}] = {}\n", i, j, model.eval(z[getPrefetchVertexIndex(i, j)]).get_decimal_string(6));
    // };
    // printZPrefetch(0, 0);
    // printZPrefetch(0, 1);
    // printZPrefetch(0, 2);
  } else {
    fmt::print("No solution found.\n");
  }
}

int main() {
  // iteExample2();
  // longestPathExample();
  chainOfStreamKernelsExample();
  return 0;
}
