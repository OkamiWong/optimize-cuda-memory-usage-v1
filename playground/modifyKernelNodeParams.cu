#include <cuda.h>

#include <cstdio>

#include "../utilities/cudaGraphUtilities.hpp"
#include "../utilities/cudaUtilities.hpp"

__global__ void foo(int8_t a, int16_t b, int32_t c, int64_t d, int *p) {
  printf("a = %d\n", a);
  printf("b = %d\n", b);
  printf("c = %d\n", c);
  printf("d = %lld\n", d);
  printf("*p = %d\n", *p);
}

void ModifyKernelNodeParams(cudaGraphNode_t kernelNode) {
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

int main() {
  int *p;
  checkCudaErrors(cudaMallocManaged(&p, sizeof(int)));
  *p = 233;

  printf("p = %p\n", p);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

  foo<<<1, 1, 0, stream>>>(101, 102, 103, 104, p);

  cudaGraph_t graph;
  checkCudaErrors(cudaStreamEndCapture(stream, &graph));

  checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", cudaGraphDebugDotFlagsVerbose));

  ModifyKernelNodeParams(getRootNode(graph));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  return 0;
}
