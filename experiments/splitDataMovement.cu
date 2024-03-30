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
#include "memopt.hpp"

using namespace memopt;

constexpr int REPETITION = 100;
constexpr int PCIE_DEVICE_ID = 0;
constexpr int NVLINK_DEVICE_ID_A = 1;
constexpr int NVLINK_DEVICE_ID_B = 2;

constexpr int MAX_SPLIT = 16;

void printHeader() {
  std::stringstream ss;
  auto csvWriter = csv::make_csv_writer(ss);
  csvWriter << std::make_tuple("kind", "split", "size(Byte)", "time(s)", "speed(GB/s)");
  fputs(ss.str().c_str(), stdout);
}

void printDataOfTheSameKind(const std::string &kind, const std::vector<size_t> &sizes, const int split, const std::vector<float> &times) {
  std::stringstream ss;
  auto csvWriter = csv::make_csv_writer(ss);
  for (int i = 0; i < sizes.size(); i++) {
    csvWriter << std::make_tuple(
      kind,
      split,
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
__global__ void initializeArrayKernel(T *array, T initialValue, size_t count) {
  const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < count) {
    array[i] = initialValue;
  }
}

void testNvlinkBandwidth(const std::vector<size_t> &sizes, int split, bool useUnifiedMemory, bool noHeader) {
  enablePeerAccessForNvlink();

  cudaStream_t streams[MAX_SPLIT];
  checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
  for (int i = 0; i < split; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  std::vector<float> deviceToDeviceTimes;

  for (auto size : sizes) {
    assert(size % split == 0);
    const size_t singleArraySize = size / split;

    float minDeviceToDeviceTime = std::numeric_limits<float>::max();

    if (useUnifiedMemory) {
      int *array[MAX_SPLIT];
      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaMallocManaged(&array[i], singleArraySize));
        initializeArrayKernel<<<(singleArraySize / 512) + 1, 512>>>(array[i], 0, singleArraySize / sizeof(int));
      }
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
      CudaEventClock clock;
      for (int i = 0; i < REPETITION; i++) {
        clock.start();
        for (int j = 0; j < split; j++) {
          checkCudaErrors(cudaMemPrefetchAsync(array[j], singleArraySize, i % 2 == 0 ? NVLINK_DEVICE_ID_B : NVLINK_DEVICE_ID_A, streams[j]));
        }
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());

        minDeviceToDeviceTime = std::min(minDeviceToDeviceTime, clock.getTimeInSeconds());
      }

      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaFree(array[i]));
      }

      deviceToDeviceTimes.push_back(minDeviceToDeviceTime);
    } else {
      int *arrayOnDeviceA[MAX_SPLIT];
      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
      for (int i = 0; i < split; i++) {
        cudaMalloc(&arrayOnDeviceA[i], singleArraySize);
        initializeArrayKernel<<<(singleArraySize / 512) + 1, 512>>>(arrayOnDeviceA[i], 0, singleArraySize / sizeof(int));
      }
      checkCudaErrors(cudaDeviceSynchronize());

      int *arrayOnDeviceB[MAX_SPLIT];
      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_B));
      for (int i = 0; i < split; i++) {
        cudaMalloc(&arrayOnDeviceB[i], singleArraySize);
        initializeArrayKernel<<<(singleArraySize / 512) + 1, 512>>>(arrayOnDeviceB[i], 0, singleArraySize / sizeof(int));
      }
      checkCudaErrors(cudaDeviceSynchronize());

      checkCudaErrors(cudaSetDevice(NVLINK_DEVICE_ID_A));
      CudaEventClock clock;
      for (int i = 0; i < REPETITION; i++) {
        clock.start();
        for (int j = 0; j < split; j++) {
          checkCudaErrors(cudaMemcpyAsync(arrayOnDeviceA[j], arrayOnDeviceB[j], singleArraySize, cudaMemcpyDeviceToDevice, streams[j]));
        }
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        minDeviceToDeviceTime = std::min(minDeviceToDeviceTime, clock.getTimeInSeconds());
      }

      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaFree(arrayOnDeviceA[i]));
        checkCudaErrors(cudaFree(arrayOnDeviceB[i]));
      }

      deviceToDeviceTimes.push_back(minDeviceToDeviceTime);
    }
  }

  for (int i = 0; i < split; i++) {
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }

  disablePeerAccessForNvlink();

  if (!noHeader) {
    printHeader();
  }

  if (useUnifiedMemory) {
    printDataOfTheSameKind("NVLink-DeviceToDevice-UnifiedMemory-Split", sizes, split, deviceToDeviceTimes);
  } else {
    printDataOfTheSameKind("NVLink-DeviceToDevice-Split", sizes, split, deviceToDeviceTimes);
  }
}

void testPcieBandwidth(const std::vector<size_t> &sizes, int split, bool useUnifiedMemory, bool noHeader) {
  std::vector<float> hostToDeviceTimes, deviceToHostTimes;

  warmUpCudaDevice();

  cudaStream_t streams[MAX_SPLIT];
  for (int i = 0; i < split; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  for (auto size : sizes) {
    assert(size % split == 0);
    const size_t singleArraySize = size / split;

    float minHostToDeviceTime = std::numeric_limits<float>::max();
    float minDeviceToHostTime = std::numeric_limits<float>::max();

    if (useUnifiedMemory) {
      int *array[MAX_SPLIT];
      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaMallocManaged(&array[i], singleArraySize));
        memset(array[i], 0, singleArraySize);
      }

      CudaEventClock clock;

      for (int i = 0; i < REPETITION; i++) {
        clock.start();
        for (int j = 0; j < split; j++) {
          checkCudaErrors(cudaMemPrefetchAsync(array[j], singleArraySize, PCIE_DEVICE_ID, streams[j]));
        }
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        minHostToDeviceTime = std::min(minHostToDeviceTime, clock.getTimeInSeconds());

        clock.start();
        for (int j = 0; j < split; j++) {
          checkCudaErrors(cudaMemPrefetchAsync(array[j], singleArraySize, cudaCpuDeviceId, streams[j]));
        }
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        minDeviceToHostTime = std::min(minDeviceToHostTime, clock.getTimeInSeconds());
      }

      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaFree(array[i]));
      }

      hostToDeviceTimes.push_back(minHostToDeviceTime);
      deviceToHostTimes.push_back(minDeviceToHostTime);
    } else {
      int *hostArray[MAX_SPLIT];
      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaMallocHost(&hostArray[i], singleArraySize));
        memset(hostArray[i], 0, singleArraySize);
      }

      int *deviceArray[MAX_SPLIT];
      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaMalloc(&deviceArray[i], singleArraySize));
      }

      CudaEventClock clock;
      for (int i = 0; i < REPETITION; i++) {
        clock.start();
        for (int j = 0; j < split; j++) {
          checkCudaErrors(cudaMemcpyAsync(deviceArray[j], hostArray[j], singleArraySize, cudaMemcpyHostToDevice, streams[j]));
        }
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        minHostToDeviceTime = std::min(minHostToDeviceTime, clock.getTimeInSeconds());

        clock.start();
        for (int j = 0; j < split; j++) {
          checkCudaErrors(cudaMemcpyAsync(hostArray[j], deviceArray[j], singleArraySize, cudaMemcpyDeviceToHost, streams[j]));
        }
        clock.end();
        checkCudaErrors(cudaDeviceSynchronize());
        minDeviceToHostTime = std::min(minDeviceToHostTime, clock.getTimeInSeconds());
      }

      for (int i = 0; i < split; i++) {
        checkCudaErrors(cudaFreeHost(hostArray[i]));
        checkCudaErrors(cudaFree(deviceArray[i]));
      }

      hostToDeviceTimes.push_back(minHostToDeviceTime);
      deviceToHostTimes.push_back(minDeviceToHostTime);
    }
  }

  for (int i = 0; i < split; i++) {
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }

  if (!noHeader) {
    printHeader();
  }

  if (useUnifiedMemory) {
    printDataOfTheSameKind("PCIe-HostToDevice-UnifiedMemory-Split", sizes, split, hostToDeviceTimes);
    printDataOfTheSameKind("PCIe-DeviceToHost-UnifiedMemory-Split", sizes, split, deviceToHostTimes);
  } else {
    printDataOfTheSameKind("PCIe-HostToDevice-Split", sizes, split, hostToDeviceTimes);
    printDataOfTheSameKind("PCIe-DeviceToHost-Split", sizes, split, deviceToHostTimes);
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

  int split;
  cmdl("split", 2) >> split;  // Split the data into two arrays and enqueue them into one stream by default
  assert(split < MAX_SPLIT);

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
    testNvlinkBandwidth(sizes, split, useUnifiedMemory, noHeader);
  } else {
    testPcieBandwidth(sizes, split, useUnifiedMemory, noHeader);
  }

  return 0;
}