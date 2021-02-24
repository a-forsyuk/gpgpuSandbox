#include "Map.cuh"

__host__ Map::Map(float2 leftUpperWorldCorner, float2 rightBottomWorldCorner, int xNodesCount, int yNodesCount):
	_luCorner(leftUpperWorldCorner),
	_rbCorner(rightBottomWorldCorner),
	_worldWidth(rightBottomWorldCorner.x - leftUpperWorldCorner.x),
	_worldHeight(rightBottomWorldCorner.y - leftUpperWorldCorner.y),
	_xNodesCount(xNodesCount),
	_yNodesCount(yNodesCount)
{
	_nodeWidth = _worldWidth/xNodesCount;
	_nodeHeight = _worldHeight/yNodesCount;
}

__host__ Map::~Map(void)
{
}

__host__ Map* Map::GetMap()
{
	return _mapSingleton;
}

__host__ void Map::Initialize(float2 leftUpperWorldCorner, float2 rightBottomWorldCorner, int xNodesCount, int yNodesCount, int agentCount)
{
	_mapSingleton = new Map(leftUpperWorldCorner, rightBottomWorldCorner, xNodesCount, yNodesCount);
}

__host__ void Map::Deinitialize()
{
	delete _mapSingleton;
}

int Map::HeightNodesCount()
{
	return GetMap()->_yNodesCount;
}

int Map::WidthNodesCount()
{
	return GetMap()->_xNodesCount;
}

__host__ __device__ int Map::GetNodeId( float x, float y )
{
	float xf = GetNodeX(x);
	float yf = GetNodeY(y);
	int result = static_cast<int>(xf)+static_cast<int>(yf)*WidthNodesCount();
	return result;
}

__host__ __device__ int Map::GetNodeX(float x)
{
	return x/GetMap()->_nodeWidth;
}
__host__ __device__ int Map::GetNodeY(float y)
{
	return y/GetMap()->_nodeHeight;
}

Map* Map::_mapSingleton = NULL;
