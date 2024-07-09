add_library(utilities STATIC utilities.cpp)

add_library(cudaUtilities STATIC cudaUtilities.cu)

add_library(cudaGraphUtilities STATIC cudaGraphUtilities.cu)
target_link_libraries(cudaGraphUtilities PRIVATE cudaUtilities CUDA::cuda_driver)
