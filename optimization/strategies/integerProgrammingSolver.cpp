#include "integerProgrammingSolver.hpp"

#include <ortools/constraint_solver/routing.h>
#include <ortools/constraint_solver/routing_enums.pb.h>
#include <ortools/constraint_solver/routing_index_manager.h>
#include <ortools/constraint_solver/routing_parameters.h>

#include <utility>

#include "../../utilities/logger.hpp"

using namespace operations_research;

FirstStepSolver::FirstStepSolver(FirstStepSolver::Input &&input){
  this->input = std::move(input);
}

FirstStepSolver::Output FirstStepSolver::solve() {
  FirstStepSolver::Output output;
  return output;
}
