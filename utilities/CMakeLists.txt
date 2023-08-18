add_library(utilities SHARED utilities.cpp)

add_library(cudaUtilities SHARED cudaUtilities.cu)

add_library(peakMemoryProfiler SHARED peakMemoryProfiler.cu)
target_link_libraries(peakMemoryProfiler cudaUtilities -lcupti)
