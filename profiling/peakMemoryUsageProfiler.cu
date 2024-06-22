#include <chrono>

#include "../utilities/configurationManager.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "peakMemoryUsageProfiler.hpp"

namespace memopt {

PeakMemoryUsageProfiler::PeakMemoryUsageProfiler(int sampleIntervalMilliseconds)
    : sampleIntervalMilliseconds(sampleIntervalMilliseconds) {}

void PeakMemoryUsageProfiler::periodicallyCheckMemoryUsage() {
  checkCudaErrors(cudaSetDevice(ConfigurationManager::getConfig().mainDeviceId));

  size_t peakMemoryUsage = 0;

  size_t free, total;
  while (!this->stopFlag) {
    checkCudaErrors(cudaMemGetInfo(&free, &total));
    peakMemoryUsage = std::max(peakMemoryUsage, total - free);
    std::this_thread::sleep_for(std::chrono::milliseconds(this->sampleIntervalMilliseconds));
  }

  this->peakMemoryUsagePromise.set_value(peakMemoryUsage);
}

void PeakMemoryUsageProfiler::start() {
  this->stopFlag = false;
  this->peakMemoryUsagePromise = std::promise<size_t>();
  this->monitorThread = std::thread(&PeakMemoryUsageProfiler::periodicallyCheckMemoryUsage, this);
}

size_t PeakMemoryUsageProfiler::end() {
  this->stopFlag = true;
  this->monitorThread.join();
  return this->peakMemoryUsagePromise.get_future().get();
}

}  // namespace memopt
