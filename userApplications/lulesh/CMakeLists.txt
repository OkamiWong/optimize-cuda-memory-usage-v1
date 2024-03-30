add_executable(lulesh lulesh.cu)
target_link_libraries(
  lulesh
  memopt
  fmt::fmt
)
