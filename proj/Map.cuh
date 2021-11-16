#pragma once
#include <cuda_runtime.h>
#include "AgentGroup.cuh"

using namespace std;

class Map
{
private:
	static Map* _mapSingleton;

	float2 _luCorner;
	float2 _rbCorner;
	float _nodeWidth;
	float _nodeHeight;
	const int _xNodesCount;
	const int _yNodesCount;
	
	const float _worldWidth;
	const float _worldHeight;
	AgentGroup* _agentGroup;
	__host__ Map(float2 leftUpperWorldCorner, float2 rightBottomWorldCorner, int xNodesCount, int yNodesCount);
	__host__  ~Map(void);
public:
	static float WorldWidth() { return GetMap()->_worldWidth; }
	static float WorldHeight() { return GetMap()->_worldHeight; }

	static __host__ void Initialize(float2 leftUpperWorldCorner, float2 rightBottomWorldCorner, int xNodesCount, int yNodesCount, int agentCount);
	static __host__ Map* GetMap();
	static __host__ void Deinitialize();

	static int HeightNodesCount();
	static int WidthNodesCount();
	static __host__ __device__ int GetNodeId(float x, float y);
	static __host__ __device__ int GetNodeX(float x);
	static __host__ __device__ int GetNodeY(float y);
};

