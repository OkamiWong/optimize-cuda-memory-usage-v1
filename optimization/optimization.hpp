#pragma once

#include "executor.hpp"
#include "optimizationOutput.hpp"

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph);
float executeOptimizedGraph(OptimizationOutput& optimizedGraph, ExecuteRandomTask executeRandomTask);
