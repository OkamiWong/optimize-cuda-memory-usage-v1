#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "../optimization/optimization.hpp"
#include "../profiling/annotation.hpp"
#include "../profiling/memoryManager.hpp"
#include "../utilities/cudaUtilities.hpp"

constexpr size_t N = 512;
constexpr size_t B = 128;

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

// Restore the element order before blocks were moved to contiguous spaces.
// Set upper triangle entries (excluding diagonal entries) in column-major order to zero.
// Transpose to row-major order.
void cleanTiledCholeskyDecompositionResult(double *L, const int n, const int b) {
  auto L_copy = std::make_unique<double[]>(N * N);
  memcpy(L_copy.get(), L, N * N * sizeof(double));

  const int t = n / b;
  for (int i = 0; i < t; i++)
    for (int j = 0; j < t; j++)
      for (int k = 0; k < b; k++)
        for (int l = 0; l < b; l++)
          L[(i * b + k) + (j * b * n + l * n)] = L_copy[(b * b) * (i + j * t) + k + l * b];

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      L[i + j * n] = 0;
      std::swap(L[i + j * n], L[i * n + j]);
    }
  }
}

bool verifyCholeskyDecomposition(double *A, double *L, const int n, bool verbose = false) {
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

  if (verbose) {
    fmt::print("A:\n");
    printSquareMatrix(A, n);

    fmt::print("\nnewA:\n");
    printSquareMatrix(newA.get(), n);

    fmt::print("\nL:\n");
    printSquareMatrix(L, n);
    fmt::print("\n");

    fmt::print("error = {:.6f}\n", error);
  }

  return error <= 1e-6;
}

typedef std::pair<int, int> MatrixTile;

class TiledCholeskyGraphCreator {
 public:
  TiledCholeskyGraphCreator(cudaStream_t stream, cudaGraph_t graph) : stream(stream), graph(graph) {
    this->lastModifiedTile = std::make_pair(-1, -1);
  }
  void beginCaptureOperation(MatrixTile tileToWrite, std::initializer_list<MatrixTile> tilesToRead) {
    auto tiles = std::vector<MatrixTile>(tilesToRead);
    tiles.push_back(tileToWrite);
    auto dependencies = this->getDependencies(tiles);

    this->lastModifiedTile = tileToWrite;
    this->lastDependencies = dependencies;

    checkCudaErrors(cudaStreamBeginCaptureToGraph(this->stream, this->graph, dependencies.data(), nullptr, dependencies.size(), cudaStreamCaptureModeGlobal));
  }

  void endCaptureOperation() {
    assert(this->lastModifiedTile.first != -1 && this->lastModifiedTile.second != -1);
    checkCudaErrors(cudaStreamEndCapture(this->stream, &this->graph));
    this->tileLastModifiedByMap[this->lastModifiedTile] = this->getTailOfLastCapturedNodeChain();
    this->lastModifiedTile = std::make_pair(-1, -1);
  };

 private:
  std::map<MatrixTile, cudaGraphNode_t> tileLastModifiedByMap;
  std::map<cudaGraphNode_t, bool> visited;
  cudaStream_t stream;
  cudaGraph_t graph;
  MatrixTile lastModifiedTile;
  std::vector<cudaGraphNode_t> lastDependencies;

  std::vector<cudaGraphNode_t> getDependencies(std::vector<MatrixTile> tiles) {
    std::vector<cudaGraphNode_t> dependencies;
    for (auto tile : tiles) {
      auto it = this->tileLastModifiedByMap.find(tile);
      if (it != this->tileLastModifiedByMap.end()) {
        dependencies.push_back(it->second);
      }
    }

    auto dedupedEnd = std::unique(dependencies.begin(), dependencies.end());
    dependencies.resize(std::distance(dependencies.begin(), dedupedEnd));
    return dependencies;
  }

  cudaGraphNode_t getTailOfLastCapturedNodeChain() {
    if (lastDependencies.size() == 0) {
      size_t numEdges;
      checkCudaErrors(cudaGraphGetEdges(this->graph, nullptr, nullptr, &numEdges));
      auto from = std::make_unique<cudaGraphNode_t[]>(numEdges);
      auto to = std::make_unique<cudaGraphNode_t[]>(numEdges);
      checkCudaErrors(cudaGraphGetEdges(this->graph, from.get(), to.get(), &numEdges));

      std::map<cudaGraphNode_t, bool> hasOutGoingEdge;
      std::set<cudaGraphNode_t> noOutGoingEdgeNodes;
      for (int i = 0; i < numEdges; i++) {
        hasOutGoingEdge[from[i]] = true;
        noOutGoingEdgeNodes.erase(from[i]);
        if (!hasOutGoingEdge[to[i]])
          noOutGoingEdgeNodes.insert(to[i]);
      }

      assert(noOutGoingEdgeNodes.size() == 1);

      return *noOutGoingEdgeNodes.begin();
    } else {
      auto nodeBeforeChain = lastDependencies[0];
      size_t numDependentNodes;
      checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, nullptr, &numDependentNodes));

      assert(numDependentNodes > 0);

      auto dependentNodes = std::make_unique<cudaGraphNode_t[]>(numDependentNodes);
      checkCudaErrors(cudaGraphNodeGetDependentNodes(nodeBeforeChain, dependentNodes.get(), &numDependentNodes));

      cudaGraphNode_t chainBeginningNode;
      for (int i = 0; i < numDependentNodes; i++) {
        if (!visited[dependentNodes[i]]) {
          chainBeginningNode = dependentNodes[i];
          break;
        }
      }

      auto u = chainBeginningNode;
      while (true) {
        visited[u] = true;
        checkCudaErrors(cudaGraphNodeGetDependentNodes(u, nullptr, &numDependentNodes));
        if (numDependentNodes == 0) break;

        assert(numDependentNodes == 1);

        cudaGraphNode_t v;
        checkCudaErrors(cudaGraphNodeGetDependentNodes(u, &v, &numDependentNodes));
        u = v;
      }

      return u;
    }
  }
};

void initializeHostData(double *h_originalMatrix) {
  generateRandomSymmetricPositiveDefiniteMatrix(h_originalMatrix, N);
}

__global__ void storeBlockMatrixInContiguousSpace(double *d_matrix, double *d_originalMatrix) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t i = (idx % N) / B;
  size_t k = (idx % N) - (i * B);
  size_t j = (idx / N) / B;
  size_t l = (idx / N) - (j * B);

  if (i >= T || j >= T || k >= B || l >= B) return;

  d_matrix[(B * B) * (i + j * T) + k + l * B] = d_originalMatrix[(i * B + k) + (j * B * N + l * N)];
}

void initializeDeviceData(double *h_originalMatrix, double *d_matrix) {
  double *d_originalMatrix;
  checkCudaErrors(cudaMalloc(&d_originalMatrix, N * N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_originalMatrix, h_originalMatrix, N * N * sizeof(double), cudaMemcpyHostToDevice));

  // Reorder elements in d_matrix, such that each block matrix is stored in a contiguous space
  constexpr size_t NUM_THREADS = 1024;
  constexpr size_t NUM_BLOCKS = (N * N + NUM_THREADS) / NUM_THREADS;
  storeBlockMatrixInContiguousSpace<<<NUM_BLOCKS, NUM_THREADS>>>(d_matrix, d_originalMatrix);

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_originalMatrix));
}

void tiledCholesky(bool optimized) {
  // Initialize data
  auto h_originalMatrix = std::make_unique<double[]>(N * N);  // Column-major
  initializeHostData(h_originalMatrix.get());

  // Initialize device data
  double *d_matrix;
  checkCudaErrors(cudaMallocManaged(&d_matrix, N * N * sizeof(double)));
  initializeDeviceData(h_originalMatrix.get(), d_matrix);

  // Register matrix block addresses
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++)
      registerManagedMemoryAddress(d_matrix + (B * B) * (i + j * T), B * B * sizeof(double));

  auto getMatrixBlock = [&](int i, int j) {
    return d_matrix + (B * B) * (i + j * T);
  };

  // Initialize libraries
  cusolverDnHandle_t cusolverDnHandle;
  cusolverDnParams_t cusolverDnParams;
  cublasHandle_t cublasHandle;
  checkCudaErrors(cusolverDnCreate(&cusolverDnHandle));
  checkCudaErrors(cusolverDnCreateParams(&cusolverDnParams));
  checkCudaErrors(cublasCreate(&cublasHandle));

  // Prepare constants
  double *one, *minusOne;
  checkCudaErrors(cudaMallocManaged(&one, sizeof(double)));
  checkCudaErrors(cudaMallocManaged(&minusOne, sizeof(double)));
  *one = 1.0;
  *minusOne = -1.0;

  // Prepare buffer for potrf
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  checkCudaErrors(cusolverDnXpotrf_bufferSize(
    cusolverDnHandle,
    cusolverDnParams,
    CUBLAS_FILL_MODE_LOWER,
    B,
    CUDA_R_64F,
    d_matrix,
    B,
    CUDA_R_64F,
    &workspaceInBytesOnDevice,
    &workspaceInBytesOnHost
  ));
  void *h_workspace, *d_workspace;
  int *d_info;
  checkCudaErrors(cudaMallocManaged(&h_workspace, workspaceInBytesOnHost));
  checkCudaErrors(cudaMallocManaged(&d_workspace, workspaceInBytesOnDevice));
  checkCudaErrors(cudaMallocManaged(&d_info, sizeof(int)));

  cudaGraph_t graph;
  checkCudaErrors(cudaGraphCreate(&graph, 0));

  cudaStream_t s;
  checkCudaErrors(cudaStreamCreate(&s));

  checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, s));
  checkCudaErrors(cublasSetStream(cublasHandle, s));

  auto tiledCholeskyGraphCreator = std::make_unique<TiledCholeskyGraphCreator>(s, graph);

  for (int k = 0; k < T; k++) {
    // A[k][k] = POTRF(A[k][k])
    // L[k][k] = POTRF(A[k][k])
    tiledCholeskyGraphCreator->beginCaptureOperation(
      std::make_pair(k, k),
      {std::make_pair(k, k)}
    );
    annotateNextKernel({getMatrixBlock(k, k)}, {getMatrixBlock(k, k)}, s);
    checkCudaErrors(cusolverDnXpotrf(
      cusolverDnHandle,
      cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER,
      B,
      CUDA_R_64F,
      getMatrixBlock(k, k),
      B,
      CUDA_R_64F,
      d_workspace,
      workspaceInBytesOnDevice,
      h_workspace,
      workspaceInBytesOnHost,
      d_info
    ));
    tiledCholeskyGraphCreator->endCaptureOperation();

    for (int i = k + 1; i < T; i++) {
      // A[i][k] = TRSM(A[k][k], A[i][k])
      // L[i][k] * L[k][k]^T = A[i][k]
      tiledCholeskyGraphCreator->beginCaptureOperation(
        std::make_pair(i, k),
        {std::make_pair(k, k), std::make_pair(i, k)}
      );
      annotateNextKernel({getMatrixBlock(i, k), getMatrixBlock(k, k)}, {getMatrixBlock(i, k)}, s);
      checkCudaErrors(cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        B, B,
        one,
        getMatrixBlock(k, k), B,
        getMatrixBlock(i, k), B
      ));
      tiledCholeskyGraphCreator->endCaptureOperation();
    }

    for (int i = k + 1; i < T; i++) {
      // A[i][i] = SYRK(A[i][k], A[i][i])
      // A[i][i] = A[i][i] - L[i][k] * L[i][k]^T
      tiledCholeskyGraphCreator->beginCaptureOperation(
        std::make_pair(i, i),
        {std::make_pair(i, i), std::make_pair(i, k)}
      );
      annotateNextKernel({getMatrixBlock(i, i), getMatrixBlock(i, k)}, {getMatrixBlock(i, i)}, s);
      checkCudaErrors(cublasDsyrk(
        cublasHandle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        B, B,
        minusOne, getMatrixBlock(i, k), B,
        one, getMatrixBlock(i, i), B
      ));
      tiledCholeskyGraphCreator->endCaptureOperation();

      for (int j = i + 1; j < T; j++) {
        // A[j][i] = GEMM(A[j][k], A[i][k])
        // A[j][i] = A[j][i] - L[j][k] * L[i][k]^T
        tiledCholeskyGraphCreator->beginCaptureOperation(
          std::make_pair(j, i),
          {std::make_pair(j, i), std::make_pair(j, k), std::make_pair(i, k)}
        );
        annotateNextKernel({getMatrixBlock(j, i), getMatrixBlock(j, k), getMatrixBlock(i, k)}, {getMatrixBlock(j, i)}, s);
        checkCudaErrors(cublasGemmEx(
          cublasHandle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          B, B, B,
          minusOne,
          getMatrixBlock(j, k), CUDA_R_64F, B,
          getMatrixBlock(i, k), CUDA_R_64F, B,
          one,
          getMatrixBlock(j, i), CUDA_R_64F, B,
          CUBLAS_COMPUTE_64F,
          CUBLAS_GEMM_DEFAULT
        ));
        tiledCholeskyGraphCreator->endCaptureOperation();
      }
    }
  }

  CudaEventClock clock;

  if (optimized) {
    auto optimizedGraph = profileAndOptimize(graph);

    initializeDeviceData(h_originalMatrix.get(), d_matrix);

    clock.start();
    executeOptimizedGraph(optimizedGraph);
    clock.end();

    cleanTiledCholeskyDecompositionResult(d_matrix, N, B);
    fmt::print("Result passes verification: {}\n", verifyCholeskyDecomposition(h_originalMatrix.get(), d_matrix, N));
    fmt::print("Total time used (s): {}\n", clock.getTimeInSeconds());
  } else {
    checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.start();
    checkCudaErrors(cudaGraphLaunch(graphExec, s));
    clock.end();

    checkCudaErrors(cudaDeviceSynchronize());

    cleanTiledCholeskyDecompositionResult(d_matrix, N, B);
    fmt::print("Result passes verification: {}\n", verifyCholeskyDecomposition(h_originalMatrix.get(), d_matrix, N));
    fmt::print("Total time used (s): {}\n", clock.getTimeInSeconds());
  }

  free(h_workspace);
  cudaFree(d_matrix);
  cudaFree(d_workspace);
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);

  tiledCholesky(cmdl["optimized"]);

  return 0;
}
