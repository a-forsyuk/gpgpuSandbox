#ifndef _AGENT_KERNEL_CUH_
#define _AGENT_KERNEL_CUH_

#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include "Map.cuh"

#ifdef CUDASystems_EXPORTS
    #define CUDASystems_API __declspec(dllexport)
#else
    #define CUDASystems_API __declspec(dllimport)
#endif

AgentGroup* g_pAgentsGroup = nullptr;

float* g_pAgentsPositions = nullptr;
float* g_pAgentsColors = nullptr;
float* g_pAgentsTargets = nullptr;

unsigned g_pAgentsCount = 0;

namespace CUDASystems 
{
	void updateNonInterlievedData()
	{
		for (unsigned i = 0; i < g_pAgentsCount; i++)
		{
			Agent* agent = g_pAgentsGroup->Agents() + i;
			memcpy(g_pAgentsPositions + i * 2u, &(agent->_position), sizeof(float2));
			memcpy(g_pAgentsTargets + i * 2u, &(agent->target), sizeof(float2));
			memcpy(g_pAgentsColors + i * 4u, &(agent->color), sizeof(float3));
			g_pAgentsColors[i * 4u + 3] = 1.0f;
		}
	}

	CUDASystems_API void Init(unsigned agentsCount)
	{
		int deviceCount = 0;
		int deviceIndex = 0;
		cudaError_t status = cudaGetDeviceCount(&deviceCount);
		cudaDeviceProp deviceProp = cudaDeviceProp();
		for (int i = 0; i < deviceCount; i++)
		{
			cudaGetDeviceProperties(&deviceProp, i);
			if (deviceProp.major == 2)
				deviceIndex = i;
		}
		cudaError_t error = cudaSetDevice(deviceIndex);

		Map::Initialize(make_float2(0.0f, 0), make_float2(2000.0f, 2000.0f), 40, 40, 100);
		g_pAgentsGroup = new AgentGroup(agentsCount);

		g_pAgentsPositions = new float[agentsCount * 2u];
		memset(g_pAgentsPositions, 0, sizeof(float) * 2u * agentsCount);

		g_pAgentsTargets = new float[agentsCount * 2u];

		g_pAgentsColors = new float[agentsCount * 4u];

		g_pAgentsCount = agentsCount;

		updateNonInterlievedData();
	}

	CUDASystems_API void Release()
	{
		cudaDeviceReset();

		delete g_pAgentsGroup;

		delete[] g_pAgentsPositions;
		delete[] g_pAgentsColors;
	}

	CUDASystems_API void Update(float dt)
	{
		g_pAgentsGroup->Update(dt);

		updateNonInterlievedData();
	}

	CUDASystems_API void GetMapDimensions(float* width, float* height)
	{
		*width = Map::WorldWidth();
		*height = Map::WorldHeight();
	}

	CUDASystems_API void GetMapNodesDimensions(unsigned* width, unsigned* height)
	{
		*width = Map::WidthNodesCount();
		*height = Map::HeightNodesCount();

	}

	CUDASystems_API void MapPositions(float* data)
	{
		memcpy(data, g_pAgentsPositions, sizeof(float) * 2u * g_pAgentsCount);
	}

	CUDASystems_API void MapTargets(float* data)
	{
		memcpy(data, g_pAgentsTargets, sizeof(float) * 2u * g_pAgentsCount);
	}

	CUDASystems_API void MapColors(float* data)
	{
		memcpy(data, g_pAgentsColors, sizeof(float) * 4u * g_pAgentsCount);
	}
}

#endif