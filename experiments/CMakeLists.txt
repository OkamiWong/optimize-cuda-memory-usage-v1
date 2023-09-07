add_executable(bandwidthTest bandwidthTest.cu)
target_link_libraries(
  bandwidthTest
  utilities
  cudaUtilities
)

add_executable(splitDataMovement splitDataMovement.cu)
target_link_libraries(
  splitDataMovement
  utilities
  cudaUtilities
)

add_executable(optimizedStream optimizedStream.cu)
target_link_libraries(
  optimizedStream
  utilities
  cudaUtilities
  logger
)
