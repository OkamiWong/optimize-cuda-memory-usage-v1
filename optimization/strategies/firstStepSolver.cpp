#include "firstStepSolver.hpp"

#include <fmt/core.h>

#include "../../utilities/logger.hpp"

namespace memopt {

FirstStepSolver::FirstStepSolver(FirstStepSolver::Input &&input) {
  this->input = std::move(input);
}

FirstStepSolver::Output FirstStepSolver::solve() {
  this->visited.resize(this->input.n, false);

  this->inDegree.resize(this->input.n, 0);
  for (int u = 0; u < this->input.n; u++) {
    for (auto v : this->input.edges[u]) {
      this->inDegree[v]++;
    }
  }

  this->maxTotalOverlap = 0;
  this->currentTopologicalSort.clear();

  dfs(0);

  this->printSolution();

  return this->output;
}

void FirstStepSolver::dfs(size_t currentTotalOverlap) {
  if (this->currentTopologicalSort.size() == this->input.n) {
    // Must accept solution update to handle the situation
    // when the best total overlap is zero.
    if (currentTotalOverlap >= this->maxTotalOverlap) {
      this->maxTotalOverlap = currentTotalOverlap;
      this->output.taskGroupExecutionOrder = this->currentTopologicalSort;
    }
    return;
  }

  for (int u = 0; u < this->input.n; u++) {
    if (this->inDegree[u] != 0 || this->visited[u]) continue;

    size_t totalOverlapIncrease = 0;
    if (this->currentTopologicalSort.size() > 0) {
      totalOverlapIncrease = this->input.dataDependencyOverlapInBytes[*(this->currentTopologicalSort.rbegin())][u];
    }

    for (auto v : this->input.edges[u]) {
      this->inDegree[v]--;
    }

    this->visited[u] = true;
    this->currentTopologicalSort.push_back(u);
    dfs(currentTotalOverlap + totalOverlapIncrease);
    this->visited[u] = false;
    this->currentTopologicalSort.pop_back();

    for (auto v : this->input.edges[u]) {
      this->inDegree[v]++;
    }
  }
}

void FirstStepSolver::printSolution() {
  LOG_TRACE_WITH_INFO("Printing solution to firstStepSolver.out");

  auto fp = fopen("firstStepSolver.out", "w");

  fmt::print(fp, "maxTotalOverlap = {}\n", this->maxTotalOverlap);

  for (int i = 0; i < this->input.n; i++) {
    fmt::print(fp, "taskGroupExecutionOrder[{}] = {}\n", i, this->output.taskGroupExecutionOrder[i]);
  }

  fclose(fp);
}

}  // namespace memopt
