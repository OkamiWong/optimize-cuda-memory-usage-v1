#include <cuda.h>

#include <cassert>

#include "../../profiling/annotation.hpp"
#include "../../utilities/cudaUtilities.hpp"
#include "../taskManager.hpp"
#include "strategyUtilities.hpp"

KernelDataDependency convertKernelIOToKernelDataDependency(const KernelIO& kernelIO) {
  KernelDataDependency dep;

  for (int i = 0; i < KernelIO::MAX_NUM_PTR; i++) {
    void* ptr = kernelIO.inputs[i];

    if (ptr == nullptr) break;

    dep.inputs.push_back(std::make_tuple(ptr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
  }

  for (int i = 0; i < KernelIO::MAX_NUM_PTR; i++) {
    void* ptr = kernelIO.outputs[i];

    if (ptr == nullptr) break;

    dep.outputs.push_back(std::make_tuple(ptr, MemoryManager::managedMemoryAddressToSizeMap[ptr]));
  }

  return dep;
}

std::map<CUgraphNode, KernelDataDependency> mapKernelOntoDataDependency(
  std::vector<CUgraphNode>& nodes,
  std::map<CUgraphNode, std::vector<CUgraphNode>>& edges
) {
  const CUfunction dummyKernelHandle = TaskManager::getInstance()->getDummyKernelHandle();

  std::map<CUgraphNode, KernelDataDependency> kernelToDataDependencyMap;

  for (auto& node : nodes) {
    CUgraphNodeType nodeType;
    checkCudaErrors(cuGraphNodeGetType(node, &nodeType));
    if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS nodeParams;
      checkCudaErrors(cuGraphKernelNodeGetParams(node, &nodeParams));
      if (nodeParams.func == dummyKernelHandle) {
        assert(edges[node].size() == 1);
        auto kernelNode = edges[node][0];
        auto kernelIOPtr = reinterpret_cast<KernelIO*>(nodeParams.kernelParams[0]);
        kernelToDataDependencyMap[kernelNode] = convertKernelIOToKernelDataDependency(*kernelIOPtr);
      }
    }
  }

  return kernelToDataDependencyMap;
}
