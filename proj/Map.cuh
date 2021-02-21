#pragma once
#include "MapNode.cuh"
#include "PathResult.cuh"
#include <host_defines.h>
#include "AgentGroup.cuh"

using namespace std;

class Map
{
private:
	static Map* _mapSingleton;

	MapNode* _nodes;
	int2 _luCorner;
	int2 _rbCorner;
	float _nodeWidth;
	float _nodeHeight;
	const int _xNodesCount;
	const int _yNodesCount;
	
	const int _worldWidth;
	const int _worldHeight;
	AgentGroup* _agentGroup;
	__host__ Map(int2 leftUpperWorldCorner, int2 rightBottomWorldCorner, int xNodesCount, int yNodesCount);
	__host__  ~Map(void);
public:
	static int WorldWidth() { return GetMap()->_worldWidth; }
	static int WorldHeight() { return GetMap()->_worldHeight; }

	static __host__ void Initialize(int2 leftUpperWorldCorner, int2 rightBottomWorldCorner, int xNodesCount, int yNodesCount, int agentCount);
	static __host__ Map* GetMap();
	static __host__ void Deinitialize();

	static int HeightNodesCount();
	static int WidthNodesCount();
	__host__ __device__ void CalculatePath(int startX, int startY, int destinationX, int destionationY, PathResult** result) const;
	static __host__ __device__ int GetNodeId(float x, float y);
	static __host__ __device__ int GetNodeX(float x);
	static __host__ __device__ int GetNodeY(float y);
	static __host__ __device__ const MapNode* GetNodeAt(int x, int y);
	static __host__ __device__ bool GetPassabilityAt(int x, int y);
};

