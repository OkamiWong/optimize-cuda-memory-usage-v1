#pragma once

#include <cstdio>
#include <cstdlib>

struct CudaConstants {
  static constexpr int DEVICE_ID = 0;
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
