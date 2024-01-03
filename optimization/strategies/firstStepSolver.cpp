#include "firstStepSolver.hpp"

#include "../../utilities/logger.hpp"

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

  LOG_TRACE_WITH_INFO("maxTotalOverlap = %zu", this->maxTotalOverlap);

  return this->output;
}

void FirstStepSolver::dfs(size_t currentTotalOverlap) {
  if (this->currentTopologicalSort.size() == this->input.n) {
    if (currentTotalOverlap > this->maxTotalOverlap) {
      this->maxTotalOverlap = currentTotalOverlap;
      this->output.nodeExecutionOrder = this->currentTopologicalSort;
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
