enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

SET (SHADER_SRCS
    shaders/GeometryShader.hlsl
    shaders/Header.hlsli
    shaders/PixelShader.hlsl
    shaders/PlainVs.hlsl
    shaders/WorldVewProjectionVs.hlsl
)
set_source_files_properties(
    shaders/PlainVs.hlsl 
    PROPERTIES
        VS_SHADER_TYPE Vertex
        VS_SHADER_OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/PlainVs.h
    )
set_source_files_properties(
    shaders/GeometryShader.hlsl
    PROPERTIES
        VS_SHADER_TYPE Geometry
        VS_SHADER_OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/GeometryShader.h
    )
set_source_files_properties(
    shaders/PixelShader.hlsl
    PROPERTIES
    VS_SHADER_TYPE Pixel
    VS_SHADER_OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/PixelShader.h
    )
set_source_files_properties(
    shaders/WorldVewProjectionVs.hlsl
    PROPERTIES
        VS_SHADER_TYPE Vertex
        VS_SHADER_OUTPUT_HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/WorldVewProjectionVs.h
    )
set_source_files_properties(
    ${SHADER_SRCS}
    PROPERTIES
        VS_SHADER_DISABLE_OPTIMIZATIONS TRUE
        VS_SHADER_ENABLE_DEBUG TRUE
        VS_SHADER_MODEL 5.0
)
source_group(shaders FILES ${SHADER_SRCS})

#VS_SHADER_DISABLE_OPTIMIZATIONS
#VS_SHADER_ENTRYPOINT
#VS_SHADER_FLAGS
#VS_SHADER_OBJECT_FILE_NAME
#VS_SHADER_OUTPUT_HEADER_FILE

set(CUDA_SRCS
    Agent.cu
    Agent.cuh
	
    AgentGroup.cu
    AgentGroup.cuh
	
    AgentsKernel.cuh
	CUDASystems.cu
	
    CustomVectorOperations.cuh
	
    Map.cu
    Map.cuh
)
source_group(cuda FILES ${CUDA_SRCS})
set_source_files_properties(
    ${CUDA_SRCS}
    PROPERTIES
        COMPILE_FLAGS "-rdc=true"
)
add_library(CUDASystems SHARED ${CUDA_SRCS})
set_property(TARGET CUDASystems 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON
             )

set(CPP_SRCS

    CUDA_first.cpp
    CUDA_first.h

    stdafx.cpp
    stdafx.h

    VertexPositionColor.h
	VertexPositionColor.cpp

    AgentVertex.h
    AgentVertex.cpp

    ConstantBuffer.h

    Render.h
    Render.cpp
	
	Models.h
    Models.cpp
)
source_group(cpp FILES ${CPP_SRCS})

add_executable(launcher 
    ${CPP_SRCS}
    ${SHADER_SRCS}
)

set_target_properties( launcher
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:launcher>"
                       VS_DEBUGGER_COMMAND "$<TARGET_FILE:launcher>")
                       #VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${CMAKE_PREFIX_PATH}/bin")

#target_compile_features(launcher PUBLIC cxx_std_14) 

target_include_directories(launcher PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(launcher PUBLIC DXUT_AUTOLIB)
target_link_libraries(launcher 
    PRIVATE
    CUDASystems
    DXUT
    DXUTOpt
    #Effects11
    Usp10.lib
    d3dcompiler.lib)

if(WIN32)
    target_compile_definitions(launcher PRIVATE _UNICODE UNICODE)
    target_compile_definitions(launcher PRIVATE _WIN32_WINNT=0x0601)
endif()