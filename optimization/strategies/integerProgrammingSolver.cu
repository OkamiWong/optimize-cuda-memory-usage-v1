#include <algorithm>
#include <iterator>
#include <numeric>

#include "integerProgrammingSolver.hpp"

FirstStepSolver::Input FirstStepSolver::constructInput(OptimizationInput optimizationInput) {
  const auto n = optimizationInput.nodes.size();

  FirstStepSolver::Input firstStepInput;

  // Plus one for one extra sentinel node
  firstStepInput.dataDependencyOverlapInBytes.resize(n + 1, std::vector<size_t>(n + 1, 0));
  firstStepInput.canPrecedeInTopologicalSort.resize(n + 1, std::vector<bool>(n + 1, true));

  // Calculate canPrecedeInTopologicalSort by Floyd Algorithm
  for (int i = 0; i < n; i++) {
    firstStepInput.canPrecedeInTopologicalSort[i][i] = false;
  }

  for (const auto &[u, destinations] : optimizationInput.edges) {
    for (auto v : destinations) {
      firstStepInput.canPrecedeInTopologicalSort[v][u] = false;
    }
  }

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        firstStepInput.canPrecedeInTopologicalSort[i][j] =
          firstStepInput.canPrecedeInTopologicalSort[i][j] &&
          (firstStepInput.canPrecedeInTopologicalSort[i][k] || firstStepInput.canPrecedeInTopologicalSort[k][j]);
      }
    }
  }

  // Calculate data dependency overlaps
  for (int i = 0; i < n; i++) {
    const auto &u = optimizationInput.nodes[i];
    std::set<MemoryManager::ArrayInfo> uTotalDataDependency;
    std::set_union(
      u.dataDependency.inputs.begin(),
      u.dataDependency.inputs.end(),
      u.dataDependency.outputs.begin(),
      u.dataDependency.outputs.end(),
      std::inserter(uTotalDataDependency, uTotalDataDependency.begin())
    );

    for (int j = i + 1; j < n; j++) {
      const auto &v = optimizationInput.nodes[j];

      if (!firstStepInput.canPrecedeInTopologicalSort[i][j]) continue;

      std::set<MemoryManager::ArrayInfo> vTotalDataDependency;
      std::set_union(
        v.dataDependency.inputs.begin(),
        v.dataDependency.inputs.end(),
        v.dataDependency.outputs.begin(),
        v.dataDependency.outputs.end(),
        std::inserter(vTotalDataDependency, vTotalDataDependency.begin())
      );

      std::set<MemoryManager::ArrayInfo> dataDependencyIntersection;
      std::set_intersection(
        uTotalDataDependency.begin(),
        uTotalDataDependency.end(),
        vTotalDataDependency.begin(),
        vTotalDataDependency.end(),
        std::inserter(dataDependencyIntersection, dataDependencyIntersection.begin())
      );

      firstStepInput.dataDependencyOverlapInBytes[i][j] =
        firstStepInput.dataDependencyOverlapInBytes[j][i] =
          std::accumulate(
            dataDependencyIntersection.begin(),
            dataDependencyIntersection.end(),
            static_cast<size_t>(0),
            [](size_t a, MemoryManager::ArrayInfo b) {
              return a + std::get<1>(b);
            }
          );
    }
  }

  // Add one sentinel to convert the problem into Traveling Salesman Problem
  // {Node without outgoing edge} -> Sentinel -> {Node without incoming edge}
  std::vector<bool> hasIncomingEdge(n, false);
  for (const auto &[u, destinations] : optimizationInput.edges) {
    for (auto v : destinations) {
      hasIncomingEdge[v] = true;
    }
  }

  for (int i = 0; i < n; i++) {
    firstStepInput.canPrecedeInTopologicalSort[n][i] = !hasIncomingEdge[i];
    firstStepInput.canPrecedeInTopologicalSort[i][n] = optimizationInput.edges[i].size() == 0;
  }

  firstStepInput.canPrecedeInTopologicalSort[n][n] = false;

  return firstStepInput;
}

