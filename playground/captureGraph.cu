#include <cstdio>

#include "../utilities/cudaUtilities.hpp"

struct KernelParam {
  int a[1];
};

__global__ void foo(__grid_constant__ const KernelParam p) {
  printf("Executing foo: p = %d\n", p.a[0]);
}

__global__ void bar(int p) {
  printf("Executing bar: p = %d\n", p);
}

int main() {
  KernelParam p;
  p.a[0] = 233;

  cudaEvent_t e1, e2;
  checkCudaErrors(cudaEventCreate(&e1));
  checkCudaErrors(cudaEventCreate(&e2));

  cudaStream_t s1, s2;
  checkCudaErrors(cudaStreamCreate(&s1));
  checkCudaErrors(cudaStreamCreate(&s2));

  checkCudaErrors(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));

  bar<<<1, 1, 0, s1>>>(1);
  checkCudaErrors(cudaEventRecord(e1, s1));
  checkCudaErrors(cudaStreamWaitEvent(s2, e1));
  bar<<<1, 1, 0, s2>>>(2);
  checkCudaErrors(cudaEventRecord(e2, s2));
  checkCudaErrors(cudaStreamWaitEvent(s1, e2));
  bar<<<1, 1, 0, s1>>>(3);

  cudaGraph_t graph;
  checkCudaErrors(cudaStreamEndCapture(s1, &graph));

  checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", cudaGraphDebugDotFlagsVerbose));

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, s1));
  checkCudaErrors(cudaDeviceSynchronize());
  return 0;
}
