#include <algorithm>
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
constexpr int DEVICE_0_ID = 0;
constexpr int DEVICE_1_ID = 1;

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

void testNvlinkBandwidth(const std::vector<size_t> &sizes) {
  // TODO
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
        checkCudaErrors(cudaMemPrefetchAsync(array, size, DEVICE_0_ID, stream));
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
  printDataOfTheSameKind("PCIe-HostToDevice", sizes, hostToDeviceTimes);
  printDataOfTheSameKind("PCIe-DeviceToHost", sizes, deviceToHostTimes);
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);

  bool useNvlink = cmdl["use-nvlink"];
  bool useUnifiedMemory = cmdl["use-unified-memory"];
  bool useLogarithmicScale = cmdl["use-log-scale"];
  bool noHeader = cmdl["no-header"];

  size_t startSize, endSize, stepSize;
  cmdl("start-size", 100'000'000ull) >> startSize;  // 100 MB
  cmdl("end-size", 1'000'000'000ull) >> endSize;     // 1 GB
  cmdl("step-size", 100'000'000ull) >> stepSize;    // 100 MB

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
    testNvlinkBandwidth(sizes);
  } else {
    testPcieBandwidth(sizes, useUnifiedMemory, noHeader);
  }

  return 0;
}