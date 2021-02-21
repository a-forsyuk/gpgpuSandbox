#ifndef _AGENT_KERNEL_CUH_
#define _AGENT_KERNEL_CUH_

#include "PathfindingKernel.cuh"
#include "SteersKernel.cuh"
texture<float4, cudaTextureType1D, cudaReadModeElementType> g_previousAgentsPositions;
texture<int4, cudaTextureType1D, cudaReadModeElementType> g_neighborsData;
texture<int2, cudaTextureType1D, cudaReadModeElementType> g_agentsHashes;
#include "Agent.cuh"
#include "AgentGroup.cuh"

#endif