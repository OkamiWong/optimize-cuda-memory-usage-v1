#include <cassert>
#include <cstdio>
#include <memory>

#include "cudaGraphExecutionTimelineProfiler.hpp"
#include "cudaUtilities.hpp"
#include "cuptiUtilities.hpp"

CudaGraphExecutionTimelineProfiler *CudaGraphExecutionTimelineProfiler::instance = nullptr;

CudaGraphExecutionTimelineProfiler *CudaGraphExecutionTimelineProfiler::getInstance() {
  if (instance == nullptr) {
    instance = new CudaGraphExecutionTimelineProfiler();
  }
  return instance;
}

void CudaGraphExecutionTimelineProfiler::consumeActivityRecord(CUpti_Activity *record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      auto kernelActivityRecord = reinterpret_cast<CUpti_ActivityKernel9 *>(record);
      this->graphNodeIdToLifetimeMap[static_cast<uint64_t>(kernelActivityRecord->graphNodeId)] =
        std::make_pair(
          static_cast<uint64_t>(kernelActivityRecord->start),
          static_cast<uint64_t>(kernelActivityRecord->end)
        );
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      auto memsetActivityRecord = reinterpret_cast<CUpti_ActivityMemset4 *>(record);
      this->graphNodeIdToLifetimeMap[static_cast<uint64_t>(memsetActivityRecord->graphNodeId)] =
        std::make_pair(
          static_cast<uint64_t>(memsetActivityRecord->start),
          static_cast<uint64_t>(memsetActivityRecord->end)
        );
      break;
    }
    default: {
      printf("Warning: Unknown CUPTI activity (%d)\n", record->kind);
      break;
    }
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  auto rawBuffer = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (rawBuffer == nullptr) {
    printf("Error: Out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = nullptr;

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      CudaGraphExecutionTimelineProfiler::getInstance()->consumeActivityRecord(record);
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      CUPTI_CALL(status);
    }
  } while (1);

  // Report any records dropped from the queue
  size_t dropped;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    printf("Dropped %u activity records\n", (unsigned int)dropped);
  }

  free(buffer);
}

void CudaGraphExecutionTimelineProfiler::graphNodeClonedCallback(CUpti_GraphData *graphData) {
  uint64_t clonedNodeId, originalNodeId;
  CUPTI_CALL(cuptiGetGraphNodeId(graphData->node, &clonedNodeId));
  CUPTI_CALL(cuptiGetGraphNodeId(graphData->originalNode, &originalNodeId));
  this->originalNodeIdToClonedNodeIdMap[originalNodeId] = clonedNodeId;
}

void CUPTIAPI
callbackHandler(
  void *pUserData,
  CUpti_CallbackDomain domain,
  CUpti_CallbackId callbackId,
  const CUpti_CallbackData *pCallbackInfo
) {
  CUPTI_CALL(cuptiGetLastError());

  switch (domain) {
    case CUPTI_CB_DOMAIN_RESOURCE: {
      CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)(pCallbackInfo);
      switch (callbackId) {
        case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED: {
          CUpti_GraphData *callbackData = (CUpti_GraphData *)(pResourceData->resourceDescriptor);
          CudaGraphExecutionTimelineProfiler::getInstance()->graphNodeClonedCallback(callbackData);
          break;
        }
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
}

void CudaGraphExecutionTimelineProfiler::initialize(cudaGraph_t graph) {
  // CUPTI Activity API
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));

  // CUPTI Callback API
  CUPTI_CALL(cuptiSubscribe(&this->subscriberHandle, (CUpti_CallbackFunc)(callbackHandler), nullptr));
  CUPTI_CALL(cuptiEnableCallback(1, this->subscriberHandle, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));

  this->finalized = false;
  this->graph = graph;
  this->graphNodeIdToLifetimeMap.clear();
  this->originalNodeIdToClonedNodeIdMap.clear();
}

void CudaGraphExecutionTimelineProfiler::finalize() {
  CUPTI_CALL(cuptiGetLastError());

  // CUPTI Activity API
  CUPTI_CALL(cuptiActivityFlushAll(1));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));

  // CUPTI Callback API
  CUPTI_CALL(cuptiUnsubscribe(this->subscriberHandle));

  this->finalized = true;
}

CudaGraphExecutionTimeline CudaGraphExecutionTimelineProfiler::getTimeline() {
  assert(this->finalized);

  size_t numNodes;
  checkCudaErrors(cudaGraphGetNodes(this->graph, nullptr, &numNodes));
  auto nodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
  checkCudaErrors(cudaGraphGetNodes(this->graph, nodes.get(), &numNodes));

  CudaGraphExecutionTimeline timeline;

  uint64_t originalNodeId;
  for (int i = 0; i < numNodes; i++) {
    CUPTI_CALL(cuptiGetGraphNodeId(nodes[i], &originalNodeId));
    timeline[nodes[i]] = this->graphNodeIdToLifetimeMap[originalNodeIdToClonedNodeIdMap[originalNodeId]];
  }

  return timeline;
}
