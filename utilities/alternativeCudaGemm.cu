#include "alternativeCudaGemm.hpp"

template <int BLOCK>
__global__ void trivialGemm(
  int m,
  int n,
  int k,
  double *alpha,
  double *a, int lda,
  double *b, int ldb,
  double *beta,
  double *c, int ldc
) {
  int _m = blockIdx.x * BLOCK + threadIdx.x;
  int _n = blockIdx.y * BLOCK + threadIdx.y;
  if (_m < m and _n < n) {
    double sum = 0.f;
    for (int i = 0; i < k; ++i) {
      sum += a[_m + i * k] * b[i * m + _n];
    }
    c[_m + _n * m] = alpha[0] * sum + beta[0] * c[_m + _n * m];
  }
}

// C = alpha * A * B^T + beta * C
void alternativeCudaGemm(
  int m,
  int n,
  int k,
  double *alpha,
  double *d_a, int lda,
  double *d_b, int ldb,
  double *beta,
  double *d_c, int ldc,
  cudaStream_t s
) {
  constexpr int BLOCK = 16;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
  trivialGemm<BLOCK><<<grid, block, 0, s>>>(m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
}
