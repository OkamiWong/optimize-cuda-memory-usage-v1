#include <z3++.h>
#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

void longestPathExample() {
  // Adjacency matrix
  std::vector<std::vector<double>> dis = {
    {0, 1, 3},
    {0, 0, 3},
    {0, 0, 0}
  };

  const int n = dis.size();

  z3::context context;
  z3::optimize optimize(context);

  std::vector<z3::expr> z;
  std::vector<std::string> zNames;

  for (int i = 0; i < n; i++) {
    zNames.push_back(fmt::format("z[{}]", i));
    if (i == 0) {
      z.push_back(context.real_val(0));
    } else {
      z.push_back(context.real_const(zNames[i].c_str()));
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

  auto handle = optimize.minimize(z[n - 1]);
  if (optimize.check() == z3::check_result::sat) {
    std::cout << optimize.lower(handle) << std::endl;
  } else {
    std::cout << "No solution found" << std::endl;
  }
}

void chainOfStreamKernelsExample() {

}

int main() {
  longestPathExample();
  chainOfStreamKernelsExample();
  return 0;
}
