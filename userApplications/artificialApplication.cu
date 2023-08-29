#include <cublas_v2.h>

#include <cassert>
#include <cstdio>

#include "../optimization/taskManager.hpp"
#include "../profiling/annotation.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"

namespace case_chainOfGemms {
template <typename T>
void fillRandomEntries(T *matrix, int m, int n, int lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i * lda + j] = 2 * static_cast<T>(drand48()) - 1;
    }
  }
}

void tf32GemmUsingTensorCore(cublasHandle_t cublasHandle, int m, int n, int k, float *d_A, float *d_B, float *d_C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  checkCudaErrors(
    cublasGemmEx(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
      d_B, CUDA_R_32F, n,
      d_A, CUDA_R_32F, k,
      &beta,
      d_C, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    )
  );
}

void runChainOfGemms(bool useGraph = true) {
  constexpr size_t CHAIN_LEN = 16;
  constexpr size_t DIMENSION = 14 * (1 << 10);

  // Calculate matrix dimensions
  const int m = DIMENSION;
  const int k = DIMENSION;
  const int n = DIMENSION;
  const size_t A_SIZE = m * k * sizeof(float);
  const size_t B_SIZE = k * n * sizeof(float);
  const size_t C_SIZE = m * n * sizeof(float);

  // Initialzie
  cublasHandle_t cublasHandle;
  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCudaErrors(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  // Allocate memory
  float *a[CHAIN_LEN], *b[CHAIN_LEN], *c[CHAIN_LEN];
  for (int i = 0; i < CHAIN_LEN; i++) {
    checkCudaErrors(cudaMallocManaged(&a[i], A_SIZE));
    checkCudaErrors(cudaMallocManaged(&b[i], B_SIZE));
    checkCudaErrors(cudaMallocManaged(&c[i], C_SIZE));
  }

  // Initialize memory
  for (int i = 0; i < CHAIN_LEN; i++) {
    fillRandomEntries(a[i], m, k, k);
    fillRandomEntries(b[i], k, n, n);
  }

  CudaEventClock clock;

  if (useGraph) {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < CHAIN_LEN; i++) {
      annotateNextKernel({a[i], b[i]}, {c[i]}, stream);
      tf32GemmUsingTensorCore(cublasHandle, m, n, k, a[i], b[i], c[i]);
    }

    checkCudaErrors(cudaGetLastError());

    cudaGraph_t graph;
    checkCudaErrors(cudaStreamEndCapture(stream, &graph));

    auto taskManager = TaskManager::getInstance();
    auto kernelRunningTimes = taskManager->getKernelRunningTimes(graph);
    for (const auto &[id, time] : kernelRunningTimes) {
      LOG_TRACE_WITH_INFO("%p: %.6f", id, time);
    }

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.start(stream);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    clock.end(stream);

    checkCudaErrors(cudaStreamDestroy(stream));
  } else {
    clock.start();
    for (int i = 0; i < CHAIN_LEN; i++) {
      tf32GemmUsingTensorCore(cublasHandle, m, n, k, a[i], b[i], c[i]);
    }
    clock.end();
  }

  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE_WITH_INFO("Total time used (s): %.2f", clock.getTimeInSeconds());

  // Clean up
  for (int i = 0; i < CHAIN_LEN; i++) {
    checkCudaErrors(cudaFree(a[i]));
    checkCudaErrors(cudaFree(b[i]));
    checkCudaErrors(cudaFree(c[i]));
  }

  checkCudaErrors(cublasDestroy(cublasHandle));
}
}  // namespace case_chainOfGemms

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
    checkCudaErrors(cudaMallocManaged(&a[i], ARRAY_SIZE));
    checkCudaErrors(cudaMallocManaged(&b[i], ARRAY_SIZE));
    checkCudaErrors(cudaMallocManaged(&c[i], ARRAY_SIZE));
  }

  // Initialize data
  for (int i = 0; i < CHAIN_LEN; i++) {
    initializeArraysKernel<<<GRID_SIZE, BLOCK_SIZE>>>(a[i], b[i], c[i], initA, initB, initC);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  CudaEventClock clock;

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  if (useGraph) {
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < CHAIN_LEN; i++) {
      annotateNextKernel({a[i], b[i]}, {c[i]}, stream);
      addKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a[i], b[i], c[i]);
    }

    checkCudaErrors(cudaGetLastError());

    cudaGraph_t graph;
    checkCudaErrors(cudaStreamEndCapture(stream, &graph));

    auto taskManager = TaskManager::getInstance();
    auto kernelRunningTimes = taskManager->getKernelRunningTimes(graph);
    for (const auto &[id, time] : kernelRunningTimes) {
      LOG_TRACE_WITH_INFO("%p: %.6f", id, time);
    }

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.start(stream);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    clock.end(stream);

    checkResultKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
      c[generateRandomInteger(0, CHAIN_LEN - 1)],
      expectedC
    );

  } else {
    clock.start();
    for (int i = 0; i < CHAIN_LEN; i++) {
      addKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a[i], b[i], c[i]);
    }
    clock.end();

    checkResultKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
      c[generateRandomInteger(0, CHAIN_LEN - 1)],
      expectedC
    );
  }

  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE_WITH_INFO("Total time used (s): %.2f", clock.getTimeInSeconds());

  // Clean up
  checkCudaErrors(cudaStreamDestroy(stream));
  for (int i = 0; i < CHAIN_LEN; i++) {
    checkCudaErrors(cudaFree(a[i]));
    checkCudaErrors(cudaFree(b[i]));
    checkCudaErrors(cudaFree(c[i]));
  }
}
}  // namespace case_chainOfStreams

int main() {
  initializeCudaDevice();

  // case_chainOfGemms::runChainOfGemms();
  case_chainOfStreams::runChainOfStreams();

  return 0;
}
