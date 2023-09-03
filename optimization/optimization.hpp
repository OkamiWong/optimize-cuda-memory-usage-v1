#pragma once

#include "customGraph.hpp"

CustomGraph profileAndOptimize(cudaGraph_t originalGraph);
void executeOptimizedGraph(CustomGraph& optimizedGraph);
