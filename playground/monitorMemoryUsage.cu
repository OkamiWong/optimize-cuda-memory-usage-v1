#include <fmt/core.h>

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

#include "../utilities/cudaUtilities.hpp"

struct PeakMemoryUsageProfiler {
  std::thread monitorThread;
  std::atomic<bool> stopFlag;
  std::promise<size_t> peakMemoryUsagePromise;

  void periodicallyCheckMemoryUsage() {
    size_t peakMemoryUsage = 0;

    size_t free, total;
    while (!stopFlag) {
      checkCudaErrors(cudaMemGetInfo(&free, &total));
      peakMemoryUsage = std::max(peakMemoryUsage, total - free);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    peakMemoryUsagePromise.set_value(peakMemoryUsage);
  }

  void start() {
    stopFlag = false;
    peakMemoryUsagePromise = std::promise<size_t>();
    monitorThread = std::thread(&PeakMemoryUsageProfiler::periodicallyCheckMemoryUsage, this);
  }

  size_t end() {
    stopFlag = true;
    monitorThread.join();
    return peakMemoryUsagePromise.get_future().get();
  }
};

float toMiB(size_t s) {
  return static_cast<float>(s) / 1024.0 / 1024.0;
}

void printMemInfo() {
  size_t free, total;
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  fmt::print(
    "free = {:.2f}\ntotal = {:.2f}\n\nused = {:.2f}\n",
    toMiB(free),
    toMiB(total),
    toMiB(total - free)
  );
}

constexpr size_t BYTES_TO_ALLOC = 1024 * 1024 * 1024;

void testCudaMemGetInfo() {
  std::string s;

  fmt::print("Start testCudaMemGetInfo\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Allocate pinned memory\n");
  void* dev_p;
  checkCudaErrors(cudaMalloc(&dev_p, BYTES_TO_ALLOC));

  fmt::print("Allocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Deallocate\n");
  checkCudaErrors(cudaFree(dev_p));

  fmt::print("Deallocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Allocate unified memory\n");
  checkCudaErrors(cudaMallocManaged(&dev_p, BYTES_TO_ALLOC));
  checkCudaErrors(cudaMemPrefetchAsync(dev_p, BYTES_TO_ALLOC, 0));

  fmt::print("Allocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Deallocate\n");
  checkCudaErrors(cudaFree(dev_p));

  fmt::print("Deallocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("End testCudaMemGetInfo\n");
}

void testPeakMemoryUsageProfiler() {
  std::string s;

  fmt::print("Start testPeakMemoryUsageProfiler\n");
  printMemInfo();
  PeakMemoryUsageProfiler profiler;
  profiler.start();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Allocate pinned memory\n");
  void* dev_p;
  checkCudaErrors(cudaMalloc(&dev_p, BYTES_TO_ALLOC));

  fmt::print("Allocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  fmt::print("Deallocate\n");
  checkCudaErrors(cudaFree(dev_p));

  fmt::print("Deallocated\n");
  printMemInfo();

  fmt::print("Press enter to continue\n");
  getline(std::cin, s);

  auto profilerResult = profiler.end();
  fmt::print("PeakMemoryUsageProfiler::end() = {:.2f}\n", toMiB(profilerResult));
}

int main() {
  testCudaMemGetInfo();

  testPeakMemoryUsageProfiler();

  return 0;
}