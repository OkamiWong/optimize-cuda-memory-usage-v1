add_executable(helloWorld helloWorld.cu)

add_executable(captureGraph captureGraph.cu)
target_link_libraries(captureGraph cudaUtilities)

add_executable(assertion assertion.cu)
