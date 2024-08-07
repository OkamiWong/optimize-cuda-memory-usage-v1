cmake_minimum_required(VERSION 3.21)

project(OptimizeCudaMemoryUsage LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CUDA_ARCHITECTURES native)
SET(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(C CXX CUDA)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(ortools REQUIRED)

add_subdirectory(utilities)
add_subdirectory(profiling)
add_subdirectory(optimization)
add_subdirectory(public)
add_subdirectory(userApplications)
add_subdirectory(playground)
add_subdirectory(experiments)
