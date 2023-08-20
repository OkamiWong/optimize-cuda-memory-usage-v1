#include <cstdio>

#include "../utilities/cudaUtilities.hpp"

struct KernelParam{
  int a[1];
};

__global__ void foo(__grid_constant__ const KernelParam p) {
  printf("Executing foo\n");
  printf("%d\n", p.a[0]);
}

int main() {
  KernelParam p;
  p.a[0] = 233;

  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));
  checkCudaErrors(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));

  foo<<<1, 1, 0, s>>>(p);

  checkCudaErrors(cudaGetLastError());

  cudaGraph_t graph;
  checkCudaErrors(cudaStreamEndCapture(s, &graph));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, s));
  checkCudaErrors(cudaDeviceSynchronize());
  return 0;
}
