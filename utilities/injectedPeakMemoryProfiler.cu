#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "logger.hpp"
#include "peakMemoryProfiler.hpp"

struct InjectionGlobals {
  volatile uint32_t initialized;
} injectionGlobals;

std::mutex initializeInjectionMutex;

void onExit() {
  LOG_TRACE();
  auto peakMemoryProfiler = PeakMemoryProfiler::getInstance();
  peakMemoryProfiler->finalize();
  printf(
    "[peakMemoryProfiler] Peak memory usage (GB): %.6lf\n",
    static_cast<double>(peakMemoryProfiler->getPeakMemoryUsage()) * 1e-9
  );
}

void initializeInjectionGlobals() {
  LOG_TRACE();
  injectionGlobals.initialized = 0;
}

void registerAtExitHandler() {
  LOG_TRACE();
  atexit(&onExit);
}

void setupCupti() {
  LOG_TRACE();
  PeakMemoryProfiler::getInstance()->initialize();
}

extern "C" int InitializeInjection() {
  if (injectionGlobals.initialized) {
    return 1;
  }

  initializeInjectionMutex.lock();

  initializeInjectionGlobals();

  registerAtExitHandler();

  setupCupti();

  injectionGlobals.initialized = 1;

  initializeInjectionMutex.unlock();

  return 1;
}