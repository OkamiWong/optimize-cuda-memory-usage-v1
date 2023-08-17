#include <cstdio>

__global__ void helloWorldFromGpu() {
  if(blockIdx.x == 0 && threadIdx.x == 0)
    printf("GPU: Hello world.\n");
}

int main() {
  printf("CPU: Hello world.\n");
  helloWorldFromGpu<<<1,1>>>();
  cudaDeviceSynchronize();
  return 0;
}