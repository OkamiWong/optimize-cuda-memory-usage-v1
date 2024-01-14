#include <cuda.h>

#include <cstdio>

#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"

struct Param {
  int8_t a; 
  int64_t c;
};

__global__ void foo(int8_t a, int16_t b, int32_t c, int64_t d, int *p, Param param, int something) {
  printf("a = %d\n", a);
  printf("b = %d\n", b);
  printf("c = %d\n", c);
  printf("d = %lld\n", d);
  printf("*p = %d\n", *p);
}

void modifyKernelNodeParams(cudaGraphNode_t kernelNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  checkCudaErrors(cuGraphKernelNodeGetParams(kernelNode, &nodeParams));

  void **kernelParams = nodeParams.kernelParams;
  printf("kernelParams = %p\n", kernelParams);

  printf("0: %zx %d\n", kernelParams[0], static_cast<int8_t *>(kernelParams[0])[0]);
  printf("1: %zx %d\n", kernelParams[1], static_cast<int16_t *>(kernelParams[1])[0]);
  printf("2: %zx %d\n", kernelParams[2], static_cast<int32_t *>(kernelParams[2])[0]);
  printf("3: %zx %lld\n", kernelParams[3], static_cast<int64_t *>(kernelParams[3])[0]);
  printf("4: %zx %zx %d\n", kernelParams[4], static_cast<int **>(kernelParams[4])[0], static_cast<int **>(kernelParams[4])[0][0]);

  printf("5: %zx\n", kernelParams[5]);
  printf("6: %zx\n", kernelParams[6]);
  printf("7: %zx\n", kernelParams[7]);
}

void countKernelNodeParams(cudaGraphNode_t kernelNode) {
  CUDA_KERNEL_NODE_PARAMS nodeParams;
  checkCudaErrors(cuGraphKernelNodeGetParams(kernelNode, &nodeParams));

  void **kernelParams = nodeParams.kernelParams;
  int count = 1;
  while (kernelParams != nullptr) {
    int offset = static_cast<char *>(kernelParams[count]) - static_cast<char *>(kernelParams[count - 1]);
    if (offset != 2 && offset != 4 && offset != 8) break;
    count++;
  }

  printf("count = %d\n", count);
}

int main() {
  int *p;
  checkCudaErrors(cudaMallocManaged(&p, sizeof(int)));
  *p = 233;

  printf("p = %p\n", p);

  Param param;

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  foo<<<1, 1, 0, stream>>>(101, 102, 103, 104, p, param, 1234);

  cudaGraph_t graph;
  checkCudaErrors(cudaStreamEndCapture(stream, &graph));

  checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", cudaGraphDebugDotFlagsVerbose));

  auto node = getRootNode(graph);
  modifyKernelNodeParams(node);
  countKernelNodeParams(node);

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  return 0;
}
