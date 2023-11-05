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

const int N = 8;

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const int n) {
  // --- Initialize random seed
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

void printSquareMatrix(double *h_A, const int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (j != 0) std::cout << " ";
      std::cout << std::setw(6) << std::setprecision(3) << h_A[i * N + j];
    }
    std::cout << std::endl;
  }
}

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

  // fmt::print("A:\n");
  // printSquareMatrix(A, n);

  // fmt::print("\nnewA:\n");
  // printSquareMatrix(newA.get(), n);

  // fmt::print("\nL:\n");
  // printSquareMatrix(L, n);
  // fmt::print("\n");

  fmt::print("error = {:.6f}\n", error);

  return error <= 1e-6;
}

int main() {
  cusolverDnHandle_t solver_handle;
  checkCudaErrors(cusolverDnCreate(&solver_handle));

  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));

  // Init
  double *h_A = (double *)malloc(N * N * sizeof(double));
  generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

  double *d_A;
  checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

  int work_size = 0;
  checkCudaErrors(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size));

  double *work;
  checkCudaErrors(cudaMalloc(&work, work_size * sizeof(double)));

  int *devInfo;
  checkCudaErrors(cudaMalloc(&devInfo, sizeof(int)));

  // Calculate
  checkCudaErrors(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, devInfo));

  // Check
  int devInfo_h = 0;
  checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) {
    std::cout << "Unsuccessful potrf execution\n\n"
              << "devInfo = " << devInfo_h << "\n\n";
  }

  // Verify
  double *h_L = (double *)malloc(N * N * sizeof(double));
  checkCudaErrors(cudaMemcpy(h_L, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
  cleanCusolverCholeskyDecompositionResult(h_L, N);
  fmt::print("Result passes verification: {}\n", verifyCholeskyDecomposition(h_A, h_L, N));

  // Clean
  checkCudaErrors(cusolverDnDestroy(solver_handle));

  return 0;
}
