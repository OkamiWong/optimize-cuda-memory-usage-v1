#include <cassert>
#include <cstdio>

#include "../optimization/optimization.hpp"
#include "../profiling/annotation.hpp"
#include "../profiling/memoryManager.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"

namespace case_chainOfStreams {
template <typename T>
__global__ void initializeArraysKernel(T *a, T *b, T *c, T initA, T initB, T initC) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <typename T>
__global__ void addKernel(const T *a, const T *b, T *c) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

template <typename T>
__global__ void checkResultKernel(const T *c, const T expectedValue) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (c[i] != expectedValue) {
    assert(false);
  }
}

void runChainOfStreams(bool useGraph = true) {
  constexpr size_t CHAIN_LEN = 16;
  constexpr size_t ARRAY_SIZE = 1 << 30;  // 1GiB
  constexpr size_t ARRAY_LEN = ARRAY_SIZE / sizeof(float);
  constexpr size_t BLOCK_SIZE = 1024;
  constexpr size_t GRID_SIZE = ARRAY_LEN / BLOCK_SIZE;

  constexpr float initA = 1;
  constexpr float initB = 2;
  constexpr float initC = 0;
  constexpr float expectedC = initA + initB;

  // Allocate memory
  float *a[CHAIN_LEN], *b[CHAIN_LEN], *c[CHAIN_LEN];
  for (int i = 0; i < CHAIN_LEN; i++) {
    checkCudaErrors(wrappedCudaMallocManaged(&a[i], ARRAY_SIZE));
    checkCudaErrors(wrappedCudaMallocManaged(&b[i], ARRAY_SIZE));
    checkCudaErrors(wrappedCudaMallocManaged(&c[i], ARRAY_SIZE));
  }

  // Initialize data
  for (int i = 0; i < CHAIN_LEN; i++) {
    initializeArraysKernel<<<GRID_SIZE, BLOCK_SIZE>>>(a[i], b[i], c[i], initA, initB, initC);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  CudaEventClock clock;

  if (useGraph) {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < CHAIN_LEN; i++) {
      annotateNextKernel({a[i], b[i]}, {c[i]}, stream);
      addKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a[i], b[i], c[i]);
    }

    checkCudaErrors(cudaGetLastError());

    cudaGraph_t graph;
    checkCudaErrors(cudaStreamEndCapture(stream, &graph));

    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaStreamDestroy(stream));

    auto optimizedGraph = profileAndOptimize(graph);

    clock.start();
    executeOptimizedGraph(optimizedGraph);
    clock.end();

    checkResultKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
      c[generateRandomInteger(0, CHAIN_LEN - 1)],
      expectedC
    );
  } else {
    clock.start();
    for (int i = 0; i < CHAIN_LEN; i++) {
      addKernel<<<GRID_SIZE, BLOCK_SIZE>>>(a[i], b[i], c[i]);
    }
    clock.end();

    checkResultKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
      c[generateRandomInteger(0, CHAIN_LEN - 1)],
      expectedC
    );
  }

  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE_WITH_INFO("Total time used (s): %.2f", clock.getTimeInSeconds());

  // Clean up
  
  for (int i = 0; i < CHAIN_LEN; i++) {
    checkCudaErrors(cudaFree(a[i]));
    checkCudaErrors(cudaFree(b[i]));
    checkCudaErrors(cudaFree(c[i]));
  }
}
}  // namespace case_chainOfStreams

int main() {
  initializeCudaDevice();

  case_chainOfStreams::runChainOfStreams();

  return 0;
}
