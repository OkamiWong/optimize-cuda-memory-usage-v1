add_executable(helloWorld helloWorld.cu)

add_executable(captureGraph captureGraph.cu)
target_link_libraries(captureGraph memopt)

add_executable(modifyKernelNodeParams modifyKernelNodeParams.cu)
target_link_libraries(modifyKernelNodeParams memopt -lcuda)

add_executable(monitorMemoryUsage monitorMemoryUsage.cu)
target_link_libraries(monitorMemoryUsage memopt fmt::fmt)

add_executable(assertion assertion.cu)

add_executable(orToolsPlayground orToolsPlayground.cpp)
target_link_libraries(orToolsPlayground ortools::ortools fmt::fmt)

add_executable(orToolsRoutingPlayground orToolsRoutingPlayground.cpp)
target_link_libraries(orToolsRoutingPlayground ortools::ortools fmt::fmt)

add_executable(tiledCholeskyPlayground tiledCholeskyPlayground.cu)
target_link_libraries(
  tiledCholeskyPlayground
  memopt
  CUDA::cublas
  CUDA::cusolver
  CUDA::curand
  fmt::fmt
)
