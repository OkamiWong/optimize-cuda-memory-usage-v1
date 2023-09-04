add_executable(bandwidthTest bandwidthTest.cu)
target_link_libraries(
  bandwidthTest
  utilities
  cudaUtilities
)
