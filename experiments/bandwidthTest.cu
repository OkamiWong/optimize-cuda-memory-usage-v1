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
#include "../utilities/utilities.hpp"

constexpr int REPETITION = 100;
constexpr int PCIE_DEVICE_ID = 0;
constexpr int NVLINK_DEVICE_ID_A = 1;
constexpr int NVLINK_DEVICE_ID_B = 2;

void printHeader() {
  std::stringstream ss;
  auto csvWriter = csv::make_csv_writer(ss);
  csvWriter << std::make_tuple("kind", "size(Byte)", "time(s)", "speed(GB/s)");
  fputs(ss.str().c_str(), stdout);
}

void printDataOfTheSameKind(const std::string &kind, const std::vector<size_t> &sizes, const std::vector<float> &times) {
  std::stringstream ss;
  auto csvWriter = csv::make_csv_writer(ss);
  for (int i = 0; i < sizes.size(); i++) {
    csvWriter << std::make_tuple(
      kind,
      sizes[i],
      toStringWithPrecision(times[i], 6),
      toStringWithPrecision(static_cast<float>(sizes[i]) / times[i] / 1e9, 3)
    );
  }
  fputs(ss.str().c_str(), stdout);
}

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
__global__ void initializeArrayKernel(T *array, T initialValue, int count) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < count) {
    array[i] = initialValue;
  }
}

void testNvlinkBandwidth(const std::vector<size_t> &sizes, bool useUnifiedMemory, bool noHeader) {
  enablePeerAccessForNvlink();

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  std::vector<float> deviceToDeviceTimes;

  for (auto size : sizes) {
    float minDeviceToDeviceTime = std::numeric_limits<float>::max();
    if (useUnifiedMemory) {
      // TODO
    } else {
      int *arrayOnDeviceA;
      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
      cudaMalloc(&arrayOnDeviceA, size);
      initializeArrayKernel<<<(size / 512) + 1, 512>>>(arrayOnDeviceA, 0, size / sizeof(int));
      checkCudaErrors(cudaDeviceSynchronize());

      int *arrayOnDeviceB;
      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_B));
      cudaMalloc(&arrayOnDeviceB, size);
      initializeArrayKernel<<<(size / 512) + 1, 512>>>(arrayOnDeviceB, 0, size / sizeof(int));
      checkCudaErrors(cudaDeviceSynchronize());

      CudaEventClock clock;
      for (int i = 0; i < REPETITION; i++) {
        clock.start(stream);
        checkCudaErrors(cudaMemcpyAsync(arrayOnDeviceA, arrayOnDeviceB, size, cudaMemcpyDeviceToDevice, stream));
        clock.end(stream);
        checkCudaErrors(cudaStreamSynchronize(stream));
        minDeviceToDeviceTime = std::min(minDeviceToDeviceTime, clock.getTimeInSeconds());
      }

      checkCudaErrors(cudaFree(arrayOnDeviceA));
      checkCudaErrors(cudaFree(arrayOnDeviceB));

      deviceToDeviceTimes.push_back(minDeviceToDeviceTime);
    }
  }

  checkCudaErrors(cudaStreamDestroy(stream));

  disablePeerAccessForNvlink();

  if (!noHeader) {
    printHeader();
  }

  if (useUnifiedMemory) {
    printDataOfTheSameKind("NVLink-DeviceToDevice-UnifiedMemory", sizes, deviceToDeviceTimes);
  } else {
    printDataOfTheSameKind("NVLink-DeviceToDevice", sizes, deviceToDeviceTimes);
  }
}

void testPcieBandwidth(const std::vector<size_t> &sizes, bool useUnifiedMemory, bool noHeader) {
  std::vector<float> hostToDeviceTimes, deviceToHostTimes;

  warmUpCudaDevice();

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  for (auto size : sizes) {
    float minHostToDeviceTime = std::numeric_limits<float>::max();
    float minDeviceToHostTime = std::numeric_limits<float>::max();

    if (useUnifiedMemory) {
      int *array;
      checkCudaErrors(cudaMallocManaged(&array, size));
      memset(array, 0, size);

      CudaEventClock clock;

      for (int i = 0; i < REPETITION; i++) {
        clock.start(stream);
        checkCudaErrors(cudaMemPrefetchAsync(array, size, PCIE_DEVICE_ID, stream));
        clock.end(stream);
        checkCudaErrors(cudaStreamSynchronize(stream));
        minHostToDeviceTime = std::min(minHostToDeviceTime, clock.getTimeInSeconds());

        clock.start(stream);
        checkCudaErrors(cudaMemPrefetchAsync(array, size, cudaCpuDeviceId));
        clock.end(stream);
        checkCudaErrors(cudaStreamSynchronize(stream));
        minDeviceToHostTime = std::min(minDeviceToHostTime, clock.getTimeInSeconds());
      }

      checkCudaErrors(cudaFree(array));

      hostToDeviceTimes.push_back(minHostToDeviceTime);
      deviceToHostTimes.push_back(minDeviceToHostTime);
    } else {
      int *hostArray;
      checkCudaErrors(cudaMallocHost(&hostArray, size));
      memset(hostArray, 0, size);

      int *deviceArray;
      checkCudaErrors(cudaMalloc(&deviceArray, size));

      CudaEventClock clock;
      for (int i = 0; i < REPETITION; i++) {
        clock.start(stream);
        checkCudaErrors(cudaMemcpyAsync(deviceArray, hostArray, size, cudaMemcpyHostToDevice, stream));
        clock.end(stream);
        checkCudaErrors(cudaStreamSynchronize(stream));
        minHostToDeviceTime = std::min(minHostToDeviceTime, clock.getTimeInSeconds());

        clock.start(stream);
        checkCudaErrors(cudaMemcpyAsync(hostArray, deviceArray, size, cudaMemcpyDeviceToHost, stream));
        clock.end(stream);
        checkCudaErrors(cudaStreamSynchronize(stream));
        minDeviceToHostTime = std::min(minDeviceToHostTime, clock.getTimeInSeconds());
      }

      checkCudaErrors(cudaFreeHost(hostArray));
      checkCudaErrors(cudaFree(deviceArray));

      hostToDeviceTimes.push_back(minHostToDeviceTime);
      deviceToHostTimes.push_back(minDeviceToHostTime);
    }
  }

  checkCudaErrors(cudaStreamDestroy(stream));

  if (!noHeader) {
    printHeader();
  }

  if (useUnifiedMemory) {
    printDataOfTheSameKind("PCIe-HostToDevice-UnifiedMemory", sizes, hostToDeviceTimes);
    printDataOfTheSameKind("PCIe-DeviceToHost-UnifiedMemory", sizes, deviceToHostTimes);
  } else {
    printDataOfTheSameKind("PCIe-HostToDevice", sizes, hostToDeviceTimes);
    printDataOfTheSameKind("PCIe-DeviceToHost", sizes, deviceToHostTimes);
  }
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);

  bool useNvlink = cmdl["use-nvlink"];
  bool useUnifiedMemory = cmdl["use-unified-memory"];
  bool useLogarithmicScale = cmdl["use-log-scale"];
  bool noHeader = cmdl["no-header"];

  size_t startSize, endSize, stepSize;
  cmdl("start-size", 10'000'000ull) >> startSize;  // 10 MB
  cmdl("end-size", 100'000'000ull) >> endSize;     // 100 MB
  cmdl("step-size", 10'000'000ull) >> stepSize;    // 10 MB

  std::vector<size_t> sizes;
  size_t s = startSize;
  while (s <= endSize) {
    sizes.push_back(s);
    if (useLogarithmicScale) {
      s *= stepSize;
    } else {
      s += stepSize;
    }
  }

  if (useNvlink) {
    testNvlinkBandwidth(sizes, useUnifiedMemory, noHeader);
  } else {
    testPcieBandwidth(sizes, useUnifiedMemory, noHeader);
  }

  return 0;
}