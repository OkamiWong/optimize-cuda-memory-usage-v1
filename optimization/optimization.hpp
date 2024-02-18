#pragma once

#include "executor.hpp"
#include "optimizationOutput.hpp"

OptimizationOutput profileAndOptimize(cudaGraph_t originalGraph);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param runningTime
///   (Output) The running time
/// @param managedDeviceArrayToHostArrayMap
///   (Output) The mapping between old managed device array addresses and
///   new host array addresses where old arrays are moved to.
void executeOptimizedGraph(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
);

/// @brief
/// @param optimizedGraph
/// @param executeRandomTask
/// @param shouldContinue
/// @param runningTime
///   (Output) The running time
/// @param managedDeviceArrayToHostArrayMap
///   (Output) The mapping between old managed device array addresses and
///   new host array addresses where old arrays are moved to.
void executeOptimizedGraphRepeatedly(
  OptimizationOutput &optimizedGraph,
  ExecuteRandomTask executeRandomTask,
  ShouldContinue shouldContinue,
  int &numIterations,
  float &runningTime,
  std::map<void *, void *> &managedDeviceArrayToHostArrayMap
);
