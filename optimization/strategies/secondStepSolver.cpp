#include "secondStepSolver.hpp"

#include "../../utilities/logger.hpp"

SecondStepSolver::SecondStepSolver(SecondStepSolver::Input &&input) {
  this->input = std::move(input);
}
