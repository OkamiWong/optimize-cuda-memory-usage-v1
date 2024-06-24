#pragma once

#include <cstdio>
#include <cstdlib>

namespace memopt {

template <typename T>
void __check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) __check((val), #val, __FILE__, __LINE__)

inline void *placeholder;

inline void reduceAvailableMemoryForUM(size_t targetSize) {
  size_t free, total;
  checkCudaErrors(cudaMemGetInfo(&free, &total));
  checkCudaErrors(cudaMalloc(&placeholder, total - targetSize));
}

inline void resetAvailableMemoryForUM() {
  checkCudaErrors(cudaFree(placeholder));
}

void warmUpCudaDevice();

void initializeCudaDevice(bool displayDeviceInfo = false);

void enablePeerAccessForNvlink(int deviceA, int deviceB);

void disablePeerAccessForNvlink(int deviceA, int deviceB);

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

}  // namespace memopt
