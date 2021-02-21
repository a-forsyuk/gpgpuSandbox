#include "PathfindingKernel.cuh"

__host__ Map::Map(int2 leftUpperWorldCorner, int2 rightBottomWorldCorner, int xNodesCount, int yNodesCount):
	_luCorner(leftUpperWorldCorner),
	_rbCorner(rightBottomWorldCorner),
	_worldWidth(rightBottomWorldCorner.x - leftUpperWorldCorner.x),
	_worldHeight(rightBottomWorldCorner.y - leftUpperWorldCorner.y),
	_xNodesCount(xNodesCount),
	_yNodesCount(yNodesCount)
{
	_nodeWidth = _worldWidth/xNodesCount;
	_nodeHeight = _worldHeight/yNodesCount;
	_nodes = new MapNode[_xNodesCount*_yNodesCount];
	for (int i=0;i<_xNodesCount;i++)
	{
		for (int j=0;j<_yNodesCount;j++)
		{
			int id = i+j*_xNodesCount;
			_nodes[id].Attrs = NodeAttributes(i,j,id);
		}
	}

	//neighborsData = new int2[xNodesCount*yNodesCount];
}

__host__ Map::~Map(void)
{
	delete [] _nodes;
}

__host__ __device__ void Map::CalculatePath(int startX, int startY, int destinationX, int destionationY, PathResult** result) const
{
	//MapOverlay* overlay = new MapOverlay(*this, GetNodeAt(startX, startY)->Attrs, GetNodeAt(destinationX, destionationY)->Attrs);
	//overlay->CalculatePath(result);
	//delete overlay;
}

__host__ __device__ const MapNode* Map::GetNodeAt( int x, int y )
{
	return &(GetMap()->_nodes[GetNodeId(x,y)]);
}

__host__ __device__ bool Map::GetPassabilityAt( int x, int y )
{
	return GetNodeAt(x, y)->IsPassable;
}

__host__ Map* Map::GetMap()
{
	return _mapSingleton;
}

__host__ void Map::Initialize(int2 leftUpperWorldCorner, int2 rightBottomWorldCorner, int xNodesCount, int yNodesCount, int agentCount)
{
	_mapSingleton = new Map(leftUpperWorldCorner, rightBottomWorldCorner, xNodesCount, yNodesCount);
	//_mapSingleton->_agentGroup = new AgentGroup(agentCount);
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
