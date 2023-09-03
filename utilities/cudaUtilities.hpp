#pragma once

#include <cstdio>
#include <cstdlib>

struct CudaConstants {
  static constexpr int DEVICE_ID = 0;
  static constexpr float PREFETCHING_BANDWIDTH_IN_GBPS = 15;
  static constexpr float PREFETCHING_BANDWIDTH = PREFETCHING_BANDWIDTH_IN_GBPS * 1e9;
};

template <typename T>
void __check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

void warmUpCudaDevice();

void initializeCudaDevice(bool displayDeviceInfo = false);

class CudaEventClock {
 public:
  CudaEventClock();
  ~CudaEventClock();
  void start(cudaStream_t stream = 0);
  void end(cudaStream_t stream = 0);
  float getTimeInSeconds();

 private:
  cudaEvent_t startEvent, endEvent;
};
