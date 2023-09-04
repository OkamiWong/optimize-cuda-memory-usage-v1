add_executable(artificialApplication artificialApplication.cu)

target_link_libraries(
  artificialApplication
  utilities
  cudaUtilities
  profiling
  optimization
  logger
)
