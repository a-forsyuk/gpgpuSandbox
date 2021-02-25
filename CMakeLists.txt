cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_steering LANGUAGES CXX CUDA)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(proj)
add_subdirectory(DXUT)
add_subdirectory(FX11-master)

SET(VS_STARTUP_PROJECT launcher)