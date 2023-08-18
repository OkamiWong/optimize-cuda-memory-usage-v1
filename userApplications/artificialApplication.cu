#include <cublas_v2.h>

#include <cstdio>

#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"

void tf32GemmUsingTensorCore(cublasHandle_t handle, int m, int n, int k, float *d_A, float *d_B, float *d_C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  checkCudaErrors(
    cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
      d_B, CUDA_R_32F, n,
      d_A, CUDA_R_32F, k,
      &beta,
      d_C, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    )
  );
}

void case_chainOfGemms() {
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
  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));
  checkCudaErrors(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

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

  clock.start();

  // Compute
  for (int i = 0; i < CHAIN_LEN; i++) {
    tf32GemmUsingTensorCore(handle, m, n, k, a[i], b[i], c[i]);
  }

  clock.end();

  checkCudaErrors(cudaDeviceSynchronize());

  LOG_TRACE_WITH_INFO("Total time used (s): %.2f", clock.getTimeInSeconds());

  // Clean up
  for (int i = 0; i < CHAIN_LEN; i++) {
    checkCudaErrors(cudaFree(a[i]));
    checkCudaErrors(cudaFree(b[i]));
    checkCudaErrors(cudaFree(c[i]));
  }

  checkCudaErrors(cublasDestroy(handle));
}

int main() {
  initializeCudaDevice();

  case_chainOfGemms();

  return 0;
}
