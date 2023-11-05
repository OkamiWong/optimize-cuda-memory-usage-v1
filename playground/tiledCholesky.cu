#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "../utilities/cudaUtilities.hpp"

const int N = 8;

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const int N) {
  // --- Initialize random seed
  srand(time(NULL));

  double *h_A_temp = (double *)malloc(N * N * sizeof(double));

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      h_A_temp[i * N + j] = (float)rand() / (float)RAND_MAX;

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      h_A[i * N + j] = 0.5 * (h_A_temp[i * N + j] + h_A_temp[j * N + i]);

  for (int i = 0; i < N; i++) h_A[i * N + i] = h_A[i * N + i] + N;
}

void printSquareMatrix(double *h_A, const int N) {
  auto originalWidth = std::cout.width();
  std::cout << std::setw(6);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (j != 0) std::cout << " ";
      std::cout << h_A[i * N + j];
    }
    std::cout << std::endl;
  }

  std::cout << std::setw(originalWidth);
}

int main() {
  cusolverDnHandle_t solver_handle;
  checkCudaErrors(cusolverDnCreate(&solver_handle));

  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));

  double *h_A = (double *)malloc(N * N * sizeof(double));
  generateRandomSymmetricPositiveDefiniteMatrix(h_A, N);

  double *d_A;
  checkCudaErrors(cudaMalloc(&d_A, N * N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));

  // cuSOLVE input/output parameters/arrays
  int work_size = 0;
  int *devInfo;
  checkCudaErrors(cudaMalloc(&devInfo, sizeof(int)));

  // CUDA CHOLESKY initialization
  checkCudaErrors(cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size));

  // CUDA POTRF execution
  double *work;
  checkCudaErrors(cudaMalloc(&work, work_size * sizeof(double)));
  checkCudaErrors(cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, devInfo));
  int devInfo_h = 0;
  checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h != 0) {
    std::cout << "Unsuccessful potrf execution\n\n"
              << "devInfo = " << devInfo_h << "\n\n";
  }

  // At this point, the lower triangular part of A contains the elements of L.
  checkCudaErrors(cudaMemcpy(h_A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost));
  printSquareMatrix(h_A, N);

  checkCudaErrors(cusolverDnDestroy(solver_handle));

  return 0;
}