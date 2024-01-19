add_executable(helloWorld helloWorld.cu)

add_executable(captureGraph captureGraph.cu)
target_link_libraries(captureGraph cudaUtilities)

add_executable(modifyKernelNodeParams modifyKernelNodeParams.cu)
target_link_libraries(modifyKernelNodeParams cudaUtilities cudaGraphUtilities -lcuda)

add_executable(assertion assertion.cu)

add_executable(orToolsPlayground orToolsPlayground.cpp)
target_link_libraries(orToolsPlayground ortools::ortools fmt::fmt)

add_executable(orToolsRoutingPlayground orToolsRoutingPlayground.cpp)
target_link_libraries(orToolsRoutingPlayground ortools::ortools fmt::fmt)

add_executable(tiledCholeskyPlayground tiledCholeskyPlayground.cu)
target_link_libraries(
  tiledCholeskyPlayground
  cudaUtilities
  annotation
  CUDA::cublas
  CUDA::cusolver
  fmt::fmt
)
