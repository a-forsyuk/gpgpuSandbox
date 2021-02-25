enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

#add_library(kernels STATIC  )

#target_compile_definitions(kernels PUBLIC __CUDACC__)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
 
add_executable(launcher 
    CUDA_first.cpp
    CUDA_first.h
    stdafx.cpp
    stdafx.h
    TransformColorInstBatch.h
    VertexPositionColor.h
    VertexPositionTexture.h

    Agent.cu
    Agent.cuh
    AgentGroup.cu
    AgentGroup.cuh
    AgentsKernel.cuh
    CustomVectorOperations.cuh
    Map.cu
    Map.cuh

    Shader.fx
)

set_target_properties( launcher
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(launcher PUBLIC cxx_std_14) 

#set_property(TARGET particle_test 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(launcher PRIVATE kernels)
target_compile_definitions(launcher PUBLIC DXUT_AUTOLIB)
target_link_libraries(launcher 
    PRIVATE
    DXUT
    DXUTOpt
    Effects11
    Usp10.lib
    d3dcompiler.lib)

if(WIN32)
    target_compile_definitions(launcher PRIVATE _UNICODE UNICODE)
    target_compile_definitions(launcher PRIVATE _WIN32_WINNT=0x0601)
endif()

add_custom_command(
    TARGET launcher POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/Shader.fx
            $<TARGET_FILE_DIR:launcher>/Shader.fx)

add_custom_command(
    TARGET launcher POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/D3D/$<IF:$<EQUAL:${CMAKE_SIZEOF_VOID_P},4>,x86,x64>/
            $<TARGET_FILE_DIR:launcher>/)