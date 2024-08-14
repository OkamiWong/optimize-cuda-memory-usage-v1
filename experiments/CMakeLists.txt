add_executable(bandwidthTest bandwidthTest.cu)
target_link_libraries(
  bandwidthTest
  memopt
)

add_executable(splitDataMovement splitDataMovement.cu)
target_link_libraries(
  splitDataMovement
  memopt
)

add_executable(optimizedStream optimizedStream.cu)
target_link_libraries(
  optimizedStream
  memopt
)

add_executable(optimizedStreamNvlink optimizedStreamNvlink.cu)
target_link_libraries(
  optimizedStreamNvlink
  memopt
)