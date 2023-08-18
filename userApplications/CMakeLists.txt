add_executable(artificialApplication artificialApplication.cu)

target_link_libraries(
  artificialApplication
  ${CUDA_cublas_LIBRARY}
  cudaUtilities
)
