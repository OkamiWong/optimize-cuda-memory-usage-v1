add_library(taskManager SHARED taskManager.cu)
target_link_libraries(
  taskManager
  -lcuda
)

add_library(optimizer SHARED optimizer.cu)
target_link_libraries(
  optimizer
  taskManager
  -lcuda
  prefetchOnlyStrategy
)

add_library(optimization SHARED optimization.cu)
target_link_libraries(
  optimization
  taskManager
  optimizer
)

add_subdirectory(strategies)