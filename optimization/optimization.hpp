#pragma once

#include "customGraph.hpp"

CustomGraph profileAndOptimize(cudaGraph_t originalGraph);
void distributeInitialData(CustomGraph& optimizedGraph);
void executeOptimizedGraph(CustomGraph& optimizedGraph);
