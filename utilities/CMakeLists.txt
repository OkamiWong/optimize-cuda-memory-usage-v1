add_library(utilities SHARED utilities.cpp)

add_library(cudaUtilities SHARED cudaUtilities.cu)

add_library(cudaGraphUtilities SHARED cudaGraphUtilities.cu)
target_link_libraries(cudaGraphUtilities cudaUtilities -lcuda)
