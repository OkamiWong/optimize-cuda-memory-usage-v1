#include <cublas_v2.h>

#include <cstdio>
#include <memory>

#include "../profiling/annotation.hpp"
#include "../utilities/cudaUtilities.hpp"
#include "../utilities/logger.hpp"
#include "../utilities/utilities.hpp"

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

void case_chainOfGemms(bool useGraph = true) {
  constexpr size_t CHAIN_LEN = 2;
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

    // Debug
    checkCudaErrors(cudaGraphDebugDotPrint(graph, "/home/twang/sources/projects/optimize-cuda-memory-usage-v1/graph.dot", cudaGraphDebugDotFlagsVerbose));
    size_t numRootNodes;
    checkCudaErrors(cudaGraphGetRootNodes(graph, nullptr, &numRootNodes));
    LOG_TRACE_WITH_INFO("%llu", numRootNodes);
    auto rootNodes = std::make_unique<cudaGraphNode_t[]>(numRootNodes);
    checkCudaErrors(cudaGraphGetRootNodes(graph, rootNodes.get(), &numRootNodes));
    cudaKernelNodeParams rootNodeParams;
    checkCudaErrors(cudaGraphKernelNodeGetParams(rootNodes[0], &rootNodeParams));
    auto io = reinterpret_cast<KernelIO *>(rootNodeParams.kernelParams[0]);
    LOG_TRACE_WITH_INFO("%p, %p", io->outputs[0], c[0]);
    // Debug END

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.start(stream);
    checkCudaErrors(cudaGraphLaunch(graphExec, stream));
    clock.end(stream);
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

int main() {
  initializeCudaDevice();

  case_chainOfGemms();

  return 0;
}
