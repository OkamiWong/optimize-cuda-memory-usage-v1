#include <fmt/core.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>

#include "../../utilities/logger.hpp"
#include "firstStepSolver.hpp"
#include "strategies.hpp"
#include "strategyUtilities.hpp"

FirstStepSolver::Input convertToFirstStepInput(OptimizationInput &optimizationInput) {
  const auto numLogicalNodes = optimizationInput.nodes.size();

  FirstStepSolver::Input firstStepInput;

  firstStepInput.n = numLogicalNodes;
  firstStepInput.edges.resize(numLogicalNodes);
  firstStepInput.dataDependencyOverlapInBytes.resize(numLogicalNodes, std::vector<size_t>(numLogicalNodes, 0));

  // Copy edges
  for (int i = 0; i < numLogicalNodes; i++) {
    firstStepInput.edges[i] = optimizationInput.edges[i];
  }

  // Calculate data dependency overlaps
  for (int i = 0; i < numLogicalNodes; i++) {
    const auto &u = optimizationInput.nodes[i];
    std::set<ArrayInfo> uTotalDataDependency;
    std::set_union(
      u.dataDependency.inputs.begin(),
      u.dataDependency.inputs.end(),
      u.dataDependency.outputs.begin(),
      u.dataDependency.outputs.end(),
      std::inserter(uTotalDataDependency, uTotalDataDependency.begin())
    );

    for (int j = i + 1; j < numLogicalNodes; j++) {
      const auto &v = optimizationInput.nodes[j];

      std::set<ArrayInfo> vTotalDataDependency;
      std::set_union(
        v.dataDependency.inputs.begin(),
        v.dataDependency.inputs.end(),
        v.dataDependency.outputs.begin(),
        v.dataDependency.outputs.end(),
        std::inserter(vTotalDataDependency, vTotalDataDependency.begin())
      );

      std::set<ArrayInfo> dataDependencyIntersection;
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
            [](size_t a, ArrayInfo b) {
              return a + std::get<1>(b);
            }
          );
    }
  }

  return firstStepInput;
}

CustomGraph TwoStepOptimizationStrategy::run(OptimizationInput &input) {
  LOG_TRACE();

  printOptimizationInput(input);

  auto firstStepInput = convertToFirstStepInput(input);
  FirstStepSolver firstStepSolver(std::move(firstStepInput));
  auto firstStepOutput = firstStepSolver.solve();

  CustomGraph optimizedGraph;
  return optimizedGraph;
}
