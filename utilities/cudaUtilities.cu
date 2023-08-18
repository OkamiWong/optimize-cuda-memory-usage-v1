#include "cudaUtilities.hpp"

__global__ void warmUp() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + static_cast<float>(tid);
}

void warmUpCudaDevice() {
  warmUp<<<32, 32>>>();
  cudaDeviceSynchronize();
}

void initializeCudaDevice(bool displayDeviceInfo) {
  checkCudaErrors(cudaSetDevice(CudaConstants::DEVICE_ID));

  if (displayDeviceInfo) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, CudaConstants::DEVICE_ID));
    printf("GPU Device %d: %s\n", CudaConstants::DEVICE_ID, deviceProp.name);
    printf("Compute Capability: %d.%d\n\n", deviceProp.major, deviceProp.minor);
  }

  warmUpCudaDevice();
}