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
);
