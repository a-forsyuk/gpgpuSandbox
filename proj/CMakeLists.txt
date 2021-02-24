enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(kernels STATIC
    Agent.cu
    Agent.cuh
    AgentGroup.cu
    AgentGroup.cuh
    AgentsKernel.cuh
    CustomVectorOperations.cuh
    Map.cu
    Map.cuh
  )
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(kernels PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( kernels
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(launcher 
    CUDA_first.cpp
    CUDA_first.h
    stdafx.cpp
    stdafx.h
    TransformColorInstBatch.h
    VertexPositionColor.h
    VertexPositionTexture.h
)
 
#set_property(TARGET particle_test 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(launcher PRIVATE kernels)
target_link_libraries(launcher PRIVATE DXUT)
target_link_libraries(launcher PRIVATE DXUTOpt)
target_link_libraries(launcher PRIVATE Effects11)