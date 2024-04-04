#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cusolverDn.h>
#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "../include/argh.h"
#include "memopt.hpp"

using namespace memopt;

size_t N;
size_t B;
size_t T;

__global__ void makeMatrixSymmetric(double *d_matrix, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t x = idx / n;
  size_t y = idx % n;

  if (x >= y || x >= n || y >= n) {
    return;
  }

  double average = 0.5 * (d_matrix[x * n + y] + d_matrix[y * n + x]);
  d_matrix[x * n + y] = average;
  d_matrix[y * n + x] = average;
}

__global__ void addIdenticalMatrix(double *d_matrix, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  d_matrix[idx * n + idx] += n;
}

// Credit to: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab
void generateRandomSymmetricPositiveDefiniteMatrix(double *h_A, const size_t n) {
  double *d_A;
  checkCudaErrors(cudaMalloc(&d_A, n * n * sizeof(double)));

  // Generate random matrix d_A
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
  curandGenerateUniformDouble(prng, d_A, n * n);

  // d_A = (d_A + d_A^T) / 2
  size_t numThreads = 1024;
  size_t numBlocks = (N * N + numThreads) / numThreads;
  makeMatrixSymmetric<<<numBlocks, numThreads>>>(d_A, N);

  // d_A = d_A + n * I
  numThreads = 1024;
  numBlocks = (N + numThreads) / numThreads;
  addIdenticalMatrix<<<numBlocks, numThreads>>>(d_A, N);

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(h_A, d_A, n * n * sizeof(double), cudaMemcpyDefault));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_A));
}

// Only verify the last row of L * L^T = A
bool verifyCholeskyDecompositionPartially(double *A, std::vector<double *> &d_tiles, const size_t n, const size_t b) {
  const size_t t = n / b;

  std::vector<std::unique_ptr<double[]>> h_tiles;
  for (int i = 0; i < t * t; i++) {
    h_tiles.push_back(std::move(std::make_unique<double[]>(b * b)));
    checkCudaErrors(cudaMemcpy(h_tiles[i].get(), d_tiles[i], b * b * sizeof(double), cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  auto getAEntry = [&](size_t row, size_t col) {
    return A[row + col * n];
  };

  auto getLEntry = [&](size_t row, size_t col) {
    if (row < col) {
      return static_cast<double>(0);
    }
    const size_t i = row / b;
    const size_t k = row - (i * b);
    const size_t j = col / b;
    const size_t l = col - (j * b);

    return h_tiles[i + j * t][k + l * b];
  };

  // Only check the last row;
  const size_t rowIndex = n - 1;

  const size_t rowLength = n;

  auto firstRow = std::make_unique<double[]>(rowLength);
  memset(firstRow.get(), 0, rowLength * sizeof(double));
  for (int j = 0; j < rowLength; j++) {
    for (int k = 0; k < n; k++) {
      firstRow[j] += getLEntry(rowIndex, k) * getLEntry(j, k);
    }
  }

  double error = 0;
  for (int j = 0; j < rowLength; j++) {
    error += fabs(getAEntry(rowIndex, j) - firstRow[j]);
  }

  fmt::print("error = {:.6f}\n", error);

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

class TiledCholeskyTaskManager {
 public:
  struct Task {
    enum class OperationType {
      portf,  // a = PORTF(a)
      trsm,   // a = a * b^T
      syrk,   // a = a - b * b^T
      gemm    // a = a - b * c^T
    };

    OperationType operation;
    MatrixTile a, b, c;
  };

  TiledCholeskyTaskManager(
    cusolverDnHandle_t cusolverDnHandle,
    cusolverDnParams_t cusolverDnParams,
    cublasHandle_t cublasHandle,
    size_t workspaceInBytesOnDevice,
    size_t workspaceInBytesOnHost,
    void *h_workspace,
    void *d_workspace,
    int *d_info,
    double *one,
    double *minusOne
  ) {
    this->cusolverDnHandle = cusolverDnHandle;
    this->cusolverDnParams = cusolverDnParams;
    this->cublasHandle = cublasHandle;
    this->workspaceInBytesOnDevice = workspaceInBytesOnDevice;
    this->workspaceInBytesOnHost = workspaceInBytesOnHost;
    this->h_workspace = h_workspace;
    this->d_workspace = d_workspace;
    this->d_info = d_info;
    this->one = one;
    this->minusOne = minusOne;
  }

  int addTask(
    Task::OperationType operation,
    MatrixTile a,
    MatrixTile b = {0, 0},
    MatrixTile c = {0, 0}
  ) {
    Task t;
    t.operation = operation;
    t.a = a;
    t.b = b;
    t.c = c;

    this->tasks.push_back(t);

    return this->tasks.size() - 1;
  }

  void executeRandomTask(std::function<double *(int, int)> getMatrixBlock, int taskId, std::map<void *, void *> addressUpdate, cudaStream_t stream) {
    const auto &task = this->tasks[taskId];

    checkCudaErrors(cusolverDnSetStream(cusolverDnHandle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    if (task.operation == Task::OperationType::portf) {
      checkCudaErrors(cusolverDnXpotrf(
        cusolverDnHandle,
        cusolverDnParams,
        CUBLAS_FILL_MODE_LOWER,
        B,
        CUDA_R_64F,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)),
        B,
        CUDA_R_64F,
        d_workspace,
        workspaceInBytesOnDevice,
        h_workspace,
        workspaceInBytesOnHost,
        d_info
      ));
    } else if (task.operation == Task::OperationType::trsm) {
      checkCudaErrors(cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        B, B,
        one,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.b.first, task.b.second)), B,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)), B
      ));
    } else if (task.operation == Task::OperationType::syrk) {
      checkCudaErrors(cublasDsyrk(
        cublasHandle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        B, B,
        minusOne, tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.b.first, task.b.second)), B,
        one, tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)), B
      ));
    } else if (task.operation == Task::OperationType::gemm) {
      checkCudaErrors(cublasGemmEx(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        B, B, B,
        minusOne,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.b.first, task.b.second)), CUDA_R_64F, B,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.c.first, task.c.second)), CUDA_R_64F, B,
        one,
        tryGettingUpdatedAddress(addressUpdate, getMatrixBlock(task.a.first, task.a.second)), CUDA_R_64F, B,
        CUBLAS_COMPUTE_64F,
        CUBLAS_GEMM_DEFAULT
      ));
    }
  }

 private:
  std::vector<Task> tasks;

  cusolverDnHandle_t cusolverDnHandle;
  cusolverDnParams_t cusolverDnParams;
  cublasHandle_t cublasHandle;
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  void *h_workspace, *d_workspace;
  int *d_info;
  double *one, *minusOne;

  template <typename T>
  T *tryGettingUpdatedAddress(std::map<void *, void *> &addressUpdate, T *oldAddress) {
    auto it = addressUpdate.find(static_cast<void *>(oldAddress));
    if (it != addressUpdate.end()) {
      return static_cast<T *>(it->second);
    }
    return oldAddress;
  }
};

void initializeHostData(double *h_originalMatrix) {
  generateRandomSymmetricPositiveDefiniteMatrix(h_originalMatrix, N);
}

__global__ void storeMatrixIntoTiles(double *d_originalMatrix, double **d_tilePointers, int N, int B, int T) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t i = (idx % N) / B;
  size_t k = (idx % N) - (i * B);
  size_t j = (idx / N) / B;
  size_t l = (idx / N) - (j * B);

  if (i >= T || j >= T || k >= B || l >= B) return;

  d_tilePointers[i + j * T][k + l * B] = d_originalMatrix[(i * B + k) + (j * B * N + l * N)];
}

void initializeDeviceData(double *h_originalMatrix, std::vector<double *> &d_tiles) {
  double *d_originalMatrix;
  checkCudaErrors(cudaMallocManaged(&d_originalMatrix, N * N * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_originalMatrix, h_originalMatrix, N * N * sizeof(double), cudaMemcpyDefault));

  double **d_tilePointers;
  checkCudaErrors(cudaMalloc(&d_tilePointers, T * T * sizeof(double *)));
  checkCudaErrors(cudaMemcpy(d_tilePointers, d_tiles.data(), T * T * sizeof(double *), cudaMemcpyDefault));

  const size_t NUM_THREADS = 1024;
  const size_t NUM_BLOCKS = (N * N + NUM_THREADS) / NUM_THREADS;
  storeMatrixIntoTiles<<<NUM_BLOCKS, NUM_THREADS>>>(
    d_originalMatrix,
    d_tilePointers,
    N, B, T
  );

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_originalMatrix));
  checkCudaErrors(cudaFree(d_tilePointers));
}

void tiledCholesky(bool optimize, bool verify) {
  SystemWallClock clock;
  clock.start();

  initializeCudaDevice();

  // Initialize data
  clock.logWithCurrentTime("Initialzing host data");
  auto h_originalMatrix = std::make_unique<double[]>(N * N);  // Column-major
  initializeHostData(h_originalMatrix.get());
  clock.logWithCurrentTime("Host data initialized");

  // Initialize device data
  clock.logWithCurrentTime("Initialzing device data");
  std::vector<double *> d_tiles(T * T);
  for (int i = 0; i < T * T; i++) {
    checkCudaErrors(cudaMalloc(&d_tiles[i], B * B * sizeof(double)));
  }
  initializeDeviceData(h_originalMatrix.get(), d_tiles);
  clock.logWithCurrentTime("Device data initialized");

  auto getMatrixBlock = [&](int i, int j) {
    return d_tiles[i + j * T];
  };

  // Register matrix block addresses
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++)
      registerManagedMemoryAddress(getMatrixBlock(i, j), B * B * sizeof(double));

  // Register application inputs and outputs
  for (int i = 0; i < T; i++) {
    for (int j = 0; j <= i; j++) {
      registerApplicationInput(getMatrixBlock(i, j));
      registerApplicationOutput(getMatrixBlock(i, j));
    }
  }
  clock.logWithCurrentTime("Addresses registered");

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
    d_tiles[0],
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

  clock.logWithCurrentTime("Preparation done, start to record graph");

  auto tiledCholeskyGraphCreator = std::make_unique<TiledCholeskyGraphCreator>(s, graph);

  auto tiledCholeskyTaskManager = std::make_unique<TiledCholeskyTaskManager>(
    cusolverDnHandle,
    cusolverDnParams,
    cublasHandle,
    workspaceInBytesOnDevice,
    workspaceInBytesOnHost,
    h_workspace,
    d_workspace,
    d_info,
    one,
    minusOne
  );

  int nextTaskId;
  for (int k = 0; k < T; k++) {
    // A[k][k] = POTRF(A[k][k])
    // L[k][k] = POTRF(A[k][k])
    tiledCholeskyGraphCreator->beginCaptureOperation(
      {k, k},
      {{k, k}}
    );
    nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::portf, {k, k});
    annotateNextTask(nextTaskId, {getMatrixBlock(k, k)}, {getMatrixBlock(k, k)}, s);
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
        {i, k},
        {{k, k}, {i, k}}
      );
      nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::trsm, {i, k}, {k, k});
      annotateNextTask(nextTaskId, {getMatrixBlock(i, k), getMatrixBlock(k, k)}, {getMatrixBlock(i, k)}, s);
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
        {i, i},
        {{i, i}, {i, k}}
      );
      nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::syrk, {i, i}, {i, k});
      annotateNextTask(nextTaskId, {getMatrixBlock(i, i), getMatrixBlock(i, k)}, {getMatrixBlock(i, i)}, s);
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
          {j, i},
          {{j, i}, {j, k}, {i, k}}
        );
        nextTaskId = tiledCholeskyTaskManager->addTask(TiledCholeskyTaskManager::Task::OperationType::gemm, {j, i}, {j, k}, {i, k});
        annotateNextTask(nextTaskId, {getMatrixBlock(j, i), getMatrixBlock(j, k), getMatrixBlock(i, k)}, {getMatrixBlock(j, i)}, s);
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

  clock.logWithCurrentTime("Graph recorded");

  LOG_TRACE_WITH_INFO("Printing original graph to graph.dot");
  checkCudaErrors(cudaGraphDebugDotPrint(graph, "./graph.dot", 0));

  clock.logWithCurrentTime("Graph printed");

  if (optimize) {
    auto optimizedGraph = profileAndOptimize(graph);

    for (int i = 0; i < ConfigurationManager::getConfig().repeat; i++) {
      initializeDeviceData(h_originalMatrix.get(), d_tiles);

      float runningTime;
      std::map<void *, void *> managedDeviceArrayToHostArrayMap;
      executeOptimizedGraph(
        optimizedGraph,
        [&](int taskId, std::map<void *, void *> addressUpdate, cudaStream_t stream) {
          tiledCholeskyTaskManager->executeRandomTask(getMatrixBlock, taskId, addressUpdate, stream);
        },
        runningTime,
        managedDeviceArrayToHostArrayMap
      );

      checkCudaErrors(cudaDeviceSynchronize());

      std::map<void *, void *> oldManagedDeviceArrayToNewManagedDeviceArrayMap;
      for (int j = 0; j < T * T; j++) {
        auto oldPtr = d_tiles[j];
        auto newPtr = managedDeviceArrayToHostArrayMap[oldPtr];
        checkCudaErrors(cudaMalloc(&d_tiles[j], B * B * sizeof(double)));
        checkCudaErrors(cudaMemcpy(d_tiles[j], newPtr, B * B * sizeof(double), cudaMemcpyDefault));
        if (ConfigurationManager::getConfig().useNvlink) {
          checkCudaErrors(cudaFree(newPtr));
        } else {
          free(newPtr);
        }
        oldManagedDeviceArrayToNewManagedDeviceArrayMap[oldPtr] = d_tiles[j];
      }

      checkCudaErrors(cudaDeviceSynchronize());

      updateManagedMemoryAddress(oldManagedDeviceArrayToNewManagedDeviceArrayMap);

      fmt::print("Total time used (s): {}\n", runningTime);
    }
  } else {
    PeakMemoryUsageProfiler peakMemoryUsageProfiler;
    CudaEventClock cudaEventClock;
    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    clock.logWithCurrentTime("Graph instantiated, start execution");

    for (int i = 0; i < ConfigurationManager::getConfig().repeat; i++) {
      initializeDeviceData(h_originalMatrix.get(), d_tiles);

      if (ConfigurationManager::getConfig().measurePeakMemoryUsage) {
        peakMemoryUsageProfiler.start();
      }

      cudaEventClock.start();
      checkCudaErrors(cudaGraphLaunch(graphExec, s));
      cudaEventClock.end();

      checkCudaErrors(cudaDeviceSynchronize());

      if (ConfigurationManager::getConfig().measurePeakMemoryUsage) {
        const auto peakMemoryUsage = peakMemoryUsageProfiler.end();
        fmt::print(
          "Peak memory usage (MiB): {:.2f}\n",
          static_cast<float>(peakMemoryUsage) / 1024.0 / 1024.0
        );
      }

      fmt::print("Total time used (s): {}\n", cudaEventClock.getTimeInSeconds());
    }
  }

  clock.logWithCurrentTime("Synchronization done");

  if (verify) {
    clock.logWithCurrentTime("Start verification");
    fmt::print("Result passes verification: {}\n", verifyCholeskyDecompositionPartially(h_originalMatrix.get(), d_tiles, N, B));
    clock.logWithCurrentTime("Verification done");
  }

  clock.logWithCurrentTime("All finished");

  free(h_workspace);
  checkCudaErrors(cudaFree(d_workspace));
  for (auto d_tile : d_tiles) {
    checkCudaErrors(cudaFree(d_tile));
  }
}

int main(int argc, char **argv) {
  auto cmdl = argh::parser(argc, argv);
  std::string configFilePath;
  cmdl("configFile", "config.json") >> configFilePath;

  ConfigurationManager::exportDefaultConfiguration();
  ConfigurationManager::loadConfiguration(configFilePath);

  N = ConfigurationManager::getConfig().tiledCholeskyN;
  T = ConfigurationManager::getConfig().tiledCholeskyT;
  B = N / T;

  tiledCholesky(
    ConfigurationManager::getConfig().optimize,
    ConfigurationManager::getConfig().verify
  );

  return 0;
}
