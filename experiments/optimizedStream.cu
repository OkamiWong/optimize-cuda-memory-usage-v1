#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "../include/csv.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"

constexpr int NVLINK_DEVICE_ID_A = 1;
constexpr int NVLINK_DEVICE_ID_B = 2;

void enablePeerAccessForNvlink() {
  int canAccessPeerAToB, canAccessPeerBToA;
  checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeerAToB, NVLINK_DEVICE_ID_A, NVLINK_DEVICE_ID_B));
  checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeerBToA, NVLINK_DEVICE_ID_B, NVLINK_DEVICE_ID_A));

  assert(canAccessPeerAToB);
  assert(canAccessPeerBToA);

  checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
  checkCudaErrors(cudaDeviceEnablePeerAccess(NVLINK_DEVICE_ID_B, 0));
  checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_B));
  checkCudaErrors(cudaDeviceEnablePeerAccess(NVLINK_DEVICE_ID_A, 0));
}

void disablePeerAccessForNvlink() {
  checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
  checkCudaErrors(cudaDeviceDisablePeerAccess(NVLINK_DEVICE_ID_B));
  checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_B));
  checkCudaErrors(cudaDeviceDisablePeerAccess(NVLINK_DEVICE_ID_A));
}

template <typename T>
__global__ void initializeArrayKernel(T *array, T initialValue, size_t count) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < count) {
    array[i] = initialValue;
  }
}

template <typename T>
__global__ void addKernel(const T *a, const T *b, T *c, size_t count) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  // if (i < count) {
  //   c[i] = a[i] + b[i];
  // }
  c[i] = a[i] + b[i];
}

template <typename T>
__global__ void checkResultKernel(const T *c, const T expectedValue) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (c[i] != expectedValue) {
    printf("[checkResultKernel] found c[%d] = %f, while expectedValue = %f\n", i, c[i], expectedValue);
  }
}

void runOptimizedStreamWithNvlink(size_t arraySize, int numberOfKernels, int prefetchCycleLength) {
  const size_t arrayLength = arraySize / sizeof(float);
  constexpr size_t BLOCK_SIZE = 1024;
  const size_t GRID_SIZE = arrayLength / BLOCK_SIZE;

  constexpr int COMPUTE_DEVICE_ID = NVLINK_DEVICE_ID_A;
  constexpr int STORAGE_DEVICE_ID = NVLINK_DEVICE_ID_B;

  constexpr float initA = 1;
  constexpr float initB = 2;
  constexpr float expectedC = initA + initB;

  enablePeerAccessForNvlink();

  // Initialize data
  auto aOnComputeDevice = std::make_unique<float *[]>(numberOfKernels);
  auto bOnComputeDevice = std::make_unique<float *[]>(numberOfKernels);
  auto cOnComputeDevice = std::make_unique<float *[]>(numberOfKernels);

  auto aOnStorageDevice = std::make_unique<float *[]>(numberOfKernels);
  auto bOnStorageDevice = std::make_unique<float *[]>(numberOfKernels);

  checkCudaErrors(cudaSetDevice(COMPUTE_DEVICE_ID));
  cudaStream_t computeStream, dataMovementStream;
  checkCudaErrors(cudaStreamCreate(&computeStream));
  checkCudaErrors(cudaStreamCreate(&dataMovementStream));

  for (int i = 0; i < numberOfKernels; i++) {
    if (i != 0 && i % prefetchCycleLength == 0) {
      checkCudaErrors(cudaSetDevice(STORAGE_DEVICE_ID));
      checkCudaErrors(cudaMalloc(&aOnStorageDevice[i], arraySize));
      checkCudaErrors(cudaMalloc(&bOnStorageDevice[i], arraySize));
      initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE>>>(aOnStorageDevice[i], initA, arrayLength);
      initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE>>>(bOnStorageDevice[i], initB, arrayLength);
      checkCudaErrors(cudaDeviceSynchronize());
    } else {
      checkCudaErrors(cudaSetDevice(COMPUTE_DEVICE_ID));
      checkCudaErrors(cudaMallocAsync(&aOnComputeDevice[i], arraySize, dataMovementStream));
      checkCudaErrors(cudaMallocAsync(&bOnComputeDevice[i], arraySize, dataMovementStream));
      initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE, 0, dataMovementStream>>>(aOnComputeDevice[i], initA, arrayLength);
      initializeArrayKernel<<<GRID_SIZE, BLOCK_SIZE, 0, dataMovementStream>>>(bOnComputeDevice[i], initB, arrayLength);
      checkCudaErrors(cudaDeviceSynchronize());
    }
  }

  // Compute
  checkCudaErrors(cudaSetDevice(COMPUTE_DEVICE_ID));

  auto prefetchEvents = std::make_unique<cudaEvent_t[]>(numberOfKernels / prefetchCycleLength);
  for (int i = 0; i < numberOfKernels / prefetchCycleLength; i++) {
    checkCudaErrors(cudaEventCreate(&prefetchEvents[i]));
  }

  CudaEventClock clock;
  clock.start(computeStream);

  for (int i = 0; i < numberOfKernels; i++) {
    if (i % prefetchCycleLength == 0) {
      if (i + prefetchCycleLength < numberOfKernels) {
        checkCudaErrors(cudaMallocAsync(&aOnComputeDevice[i + prefetchCycleLength], arraySize, dataMovementStream));
        checkCudaErrors(cudaMallocAsync(&bOnComputeDevice[i + prefetchCycleLength], arraySize, dataMovementStream));
        checkCudaErrors(cudaMemcpyAsync(aOnComputeDevice[i + prefetchCycleLength], aOnStorageDevice[i + prefetchCycleLength], arraySize, cudaMemcpyDeviceToDevice, dataMovementStream));
        checkCudaErrors(cudaMemcpyAsync(bOnComputeDevice[i + prefetchCycleLength], bOnStorageDevice[i + prefetchCycleLength], arraySize, cudaMemcpyDeviceToDevice, dataMovementStream));
        checkCudaErrors(cudaEventRecord(prefetchEvents[i / prefetchCycleLength], dataMovementStream));
      }
      if (i != 0) {
        checkCudaErrors(cudaStreamWaitEvent(computeStream, prefetchEvents[i / prefetchCycleLength - 1]));
      }
    }

    checkCudaErrors(cudaMallocAsync(&cOnComputeDevice[i], arraySize, computeStream));
    addKernel<<<GRID_SIZE, BLOCK_SIZE, 0, computeStream>>>(aOnComputeDevice[i], bOnComputeDevice[i], cOnComputeDevice[i], arrayLength);
    checkResultKernel<<<1, 1, 0, computeStream>>>(cOnComputeDevice[i], expectedC);
    checkCudaErrors(cudaFreeAsync(aOnComputeDevice[i], computeStream));
    checkCudaErrors(cudaFreeAsync(bOnComputeDevice[i], computeStream));
    checkCudaErrors(cudaFreeAsync(cOnComputeDevice[i], computeStream));
  }

  clock.end(computeStream);
  checkCudaErrors(cudaStreamSynchronize(computeStream));

  const float runningTime = clock.getTimeInSeconds();
  const float bandwidth = static_cast<float>(arraySize) * 3.0 * numberOfKernels / 1e9 / runningTime;
  LOG_TRACE_WITH_INFO("Total running time (s): %.6f", runningTime);
  LOG_TRACE_WITH_INFO("Bandwidth (GB/s): %.2f", bandwidth);

  for (int i = 0; i < numberOfKernels / prefetchCycleLength; i++) {
    checkCudaErrors(cudaEventDestroy(prefetchEvents[i]));
  }

  checkCudaErrors(cudaStreamDestroy(computeStream));
  checkCudaErrors(cudaStreamDestroy(dataMovementStream));
  disablePeerAccessForNvlink();
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);

  size_t arraySize;
  cmdl("array-size", 1'000'000'000ull) >> arraySize;  // 1GB by default

  int numberOfKernels;
  cmdl("number-of-kernels", 12) >> numberOfKernels;  // 12 kernels in total by default

  int prefetchCycleLength;
  cmdl("prefetch-cycle-length", 3) >> prefetchCycleLength;  // Prefetch the 3th, 6th, ... kernels (counting from 0th) by default

  runOptimizedStreamWithNvlink(arraySize, numberOfKernels, prefetchCycleLength);

  return 0;
}