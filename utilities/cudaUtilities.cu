#include <cassert>

#include "cudaUtilities.hpp"

namespace memopt {

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
  checkCudaErrors(cudaSetDevice(Constants::DEVICE_ID));

  if (displayDeviceInfo) {
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, Constants::DEVICE_ID));
    printf("GPU Device %d: %s\n", Constants::DEVICE_ID, deviceProp.name);
    printf("Compute Capability: %d.%d\n\n", deviceProp.major, deviceProp.minor);
  }

  warmUpCudaDevice();
}

void enablePeerAccessForNvlink(int deviceA, int deviceB) {
  int currentDevice;
  checkCudaErrors(cudaGetDevice(&currentDevice));

  int canAccessPeerAToB, canAccessPeerBToA;
  checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeerAToB, deviceA, deviceB));
  checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeerBToA, deviceB, deviceA));

  assert(canAccessPeerAToB);
  assert(canAccessPeerBToA);

  checkCudaErrors(cudaSetDevice(deviceA));
  checkCudaErrors(cudaDeviceEnablePeerAccess(deviceB, 0));
  checkCudaErrors(cudaSetDevice(deviceB));
  checkCudaErrors(cudaDeviceEnablePeerAccess(deviceA, 0));

  checkCudaErrors(cudaSetDevice(currentDevice));
}

void disablePeerAccessForNvlink(int deviceA, int deviceB) {
  int currentDevice;
  checkCudaErrors(cudaGetDevice(&currentDevice));

  checkCudaErrors(cudaSetDevice(deviceA));
  checkCudaErrors(cudaDeviceDisablePeerAccess(deviceB));
  checkCudaErrors(cudaSetDevice(deviceB));
  checkCudaErrors(cudaDeviceDisablePeerAccess(deviceA));

  checkCudaErrors(cudaSetDevice(currentDevice));
}

CudaEventClock::CudaEventClock() {
  checkCudaErrors(cudaEventCreate(&this->startEvent));
  checkCudaErrors(cudaEventCreate(&this->endEvent));
}

CudaEventClock::~CudaEventClock() {
  checkCudaErrors(cudaEventDestroy(this->startEvent));
  checkCudaErrors(cudaEventDestroy(this->endEvent));
}

void CudaEventClock::start(cudaStream_t stream) {
  checkCudaErrors(cudaEventRecord(this->startEvent, stream));
}

void CudaEventClock::end(cudaStream_t stream) {
  checkCudaErrors(cudaEventRecord(this->endEvent, stream));
}

float CudaEventClock::getTimeInSeconds() {
  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, this->startEvent, this->endEvent));
  return time * 1e-3f;
}

}  // namespace memopt
