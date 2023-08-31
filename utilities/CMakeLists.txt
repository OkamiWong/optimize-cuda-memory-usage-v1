add_library(utilities SHARED utilities.cpp)

add_library(cudaUtilities SHARED cudaUtilities.cu)

add_library(cudaGraphUtilities SHARED cudaGraphUtilities.cu)
target_link_libraries(cudaGraphUtilities cudaUtilities -lcuda)

add_library(peakMemoryProfiler SHARED peakMemoryProfiler.cu)
target_link_libraries(peakMemoryProfiler cudaUtilities -lcupti)

add_library(injectedPeakMemoryProfiler SHARED injectedPeakMemoryProfiler.cu)
target_link_libraries(injectedPeakMemoryProfiler peakMemoryProfiler)
