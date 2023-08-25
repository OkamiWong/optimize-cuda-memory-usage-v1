add_library(taskManager SHARED taskManager.cu)
target_link_libraries(
  taskManager
  -lcuda
)
