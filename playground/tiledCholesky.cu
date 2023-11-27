#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <fmt/core.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#include "../utilities/cudaUtilities.hpp"

constexpr size_t N = 8;
constexpr size_t B = 2;

constexpr size_t T = N / B;

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n) {
  srand(time(NULL));

  double *h_A_temp = (double *)malloc(n * n * sizeof(double));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      h_A_temp[i * n + j] = (float)rand() / (float)RAND_MAX;

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      h_A[i * n + j] = 0.5 * (h_A_temp[i * n + j] + h_A_temp[j * n + i]);

  for (int i = 0; i < n; i++) h_A[i * n + i] = h_A[i * n + i] + n;
}

void printSquareMatrix(double *h_A, const size_t n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (j != 0) std::cout << " ";
      std::cout << std::setw(6) << std::setprecision(3) << h_A[i * n + j];
    }
    std::cout << std::endl;
  }
}

// Set upper triangle entries (excluding diagonal entries) in column-major order to zero.
// Then, transpose to row-major order.
void cleanCusolverCholeskyDecompositionResult(double *L, const int n) {
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      L[i + j * n] = 0;
      std::swap(L[i + j * n], L[i * n + j]);
    }
  }
}

bool verifyCholeskyDecomposition(double *A, double *L, const int n) {
  auto newA = std::make_unique<double[]>(n * n);
  memset(newA.get(), 0, n * n * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        newA[i * n + j] += L[i * n + k] * L[k + j * n];
      }
    }
  }

  double error = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      error += fabs(A[i * n + j] - newA[i * n + j]);
    }
  }

  fmt::print("error = {:.6f}\n", error);

  return error <= 1e-6;
}

void trivialCholesky() {
  // Initialize libaries
  cusolverDnHandle_t cusolverDnHandle;
  checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));

  cusolverDnParams_t cusolverDnParams;
  checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));

  // Initialize data
  double *h_A = (double *)malloc(N * N * sizeof(double));
  generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

  double *d_A;
  checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;

  checkCudaErrors(cusolverDnXpotrf_bufferSize(
    cusolverDnHandle,
    cusolverDnParams,
    CUBLAS_FILL_MODE_LOWER,
    N,
    CUDA_R_64F,
    d_A,
    N,
    CUDA_R_64F,
    &workspaceInBytesOnDevice,
    &workspaceInBytesOnHost
  ));

  void *h_workspace = malloc(workspaceInBytesOnHost);

  void *d_workspace;
  checkCudaErrors(cudaMalloc(&d_workspace, workspaceInBytesOnDevice));

  int *d_info;
  checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

  // Calculate
  checkCudaErrors(cusolverDnXpotrf(
    cusolverDnHandle,
    cusolverDnParams,
    CUBLAS_FILL_MODE_LOWER,
    N,
    CUDA_R_64F,
    d_A,
    N,
    CUDA_R_64F,
    d_workspace,
    workspaceInBytesOnDevice,
    h_workspace,
    workspaceInBytesOnHost,
    d_info
  ));

  // Check
  int h_info = 0;
  checkCudaErrors(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info != 0) {
    std::cout << "Unsuccessful potrf execution\n\n"
              << "d_info = " << h_info << "\n\n";
  }

  // Verify
  double *h_L = (double *)malloc(N * N * sizeof(double));
  checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
  cleanCusolverCholeskyDecompositionResult(h_L, N);
  fmt::print("Result passes verification: {}\n", verifyCholeskyDecomposition(h_A, h_L, N));

  // Clean
  free(h_A);
  free(h_workspace);
  free(h_L);
  checkCudaErrors(cusolverDnDestroy(cusolverDnHandle));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_workspace));
  checkCudaErrors(cudaFree(d_info));
}

int main() {
  trivialCholesky();

  return 0;
}
