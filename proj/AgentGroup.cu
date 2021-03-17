#include "AgentGroup.cuh"
#include "AgentsKernel.cuh"
#include "Map.cuh"

#include <math_functions.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <texture_fetch_functions.h>

#include <device_launch_parameters.h>

extern texture<float4, cudaTextureType1D, cudaReadModeElementType> g_previousAgentsPositions;
extern texture<int4, cudaTextureType1D, cudaReadModeElementType> g_neighborsData;
extern texture<int2, cudaTextureType1D, cudaReadModeElementType> g_agentsHashes;

__host__  AgentGroup::AgentGroup(int agentsCount) :
	_agentsCount(agentsCount)
{
	cudaError_t status;
	status = cudaMallocHost(&_agents, sizeof(Agent)*_agentsCount);
	int sqrtAgentsCount = sqrt(static_cast<float>(_agentsCount));
	unsigned int actualAgentsCount = 0;

	float4* tempAgentsPosVel = new float4[agentsCount];

	for (int x = 0;;x++)
	{
		for (int y = 0;y<sqrtAgentsCount;y++)	
		{
			Agent agent;
			agent.Position(make_float2(x*20+3, y*20+3));
//			agent.Id = actualAgentsCount;
			agent.color = actualAgentsCount < agentsCount/2 ? make_float3(0.5f,0,0):make_float3(0,0.5f,0);
			agent.target = make_float2(actualAgentsCount < agentsCount/2 ? Map::WorldWidth() : 0,rand()/(32767/Map::WorldHeight())-1);
			tempAgentsPosVel[actualAgentsCount] = make_float4(agent.Position().x, agent.Position().y, agent.Velocity().x, agent.Velocity().y);
			_agents[actualAgentsCount++] = agent;
			if(actualAgentsCount>=_agentsCount)
				break;
		}
		if(actualAgentsCount>=_agentsCount)
			break;
	}

	status = cudaMalloc(&_d_agents, sizeof(Agent)*agentsCount);
	status = cudaMemcpy(_d_agents, _agents, sizeof(Agent)*agentsCount, cudaMemcpyHostToDevice);

	status = cudaMalloc(&_agentPositionVelocity, sizeof(float4)*agentsCount);
	status = cudaMemcpy(_agentPositionVelocity, tempAgentsPosVel, sizeof(float4)*agentsCount, cudaMemcpyHostToDevice);
	cudaChannelFormatDesc agentsTextureDesc = cudaCreateChannelDesc<float4>();
	status = cudaMallocArray(&_previousAgentsArray, 
		&agentsTextureDesc,
		_agentsCount,
		0);
	status = cudaMemcpyToArray(_previousAgentsArray, 0, 0, _agentPositionVelocity, sizeof(float4)*_agentsCount, cudaMemcpyDeviceToDevice);
	//g_obstaclesTexture.normalized = false;
	status = cudaBindTextureToArray(&g_previousAgentsPositions, _previousAgentsArray, &agentsTextureDesc);

	thrust::host_vector<int2> tempAgentsHashes = thrust::host_vector<int2>();
	for (int i = 0; i<_agentsCount;i++)
	{
		tempAgentsHashes.push_back(make_int2(i, 0));
	}
	_agentsHashes = tempAgentsHashes;
	cudaChannelFormatDesc hashesTextureDesc = cudaCreateChannelDesc<int2>();
	status = cudaMallocArray(&_agentsHashesArray, 
		&hashesTextureDesc,
		_agentsCount,
		0);
	status = cudaMemcpyToArray(_agentsHashesArray, 0, 0, thrust::raw_pointer_cast(&_agentsHashes[0]), sizeof(int2)*_agentsCount, cudaMemcpyDeviceToDevice);
	g_agentsHashes.normalized = false;
	status = cudaBindTextureToArray(&g_agentsHashes, _agentsHashesArray, &hashesTextureDesc);


	thrust::host_vector<int4> tempNeighborsData = thrust::host_vector<int4>();
	for (int i=0;i<Map::WidthNodesCount()*Map::HeightNodesCount();i++)
	{
		tempNeighborsData.push_back(make_int4(i, 0, 0, 0));
	}
	_neighborsData = tempNeighborsData;
	cudaChannelFormatDesc neighborsTextureDesc = cudaCreateChannelDesc<int4>();
	status = cudaMallocArray(&_neighborsDataArray, 
		&neighborsTextureDesc,
		_neighborsData.size(),
		0);
	status = cudaMemcpyToArray(_neighborsDataArray, 0, 0, thrust::raw_pointer_cast(&_neighborsData[0]), sizeof(int4)*_neighborsData.size(), cudaMemcpyDeviceToDevice);
	g_neighborsData.normalized = false;
	status = cudaBindTextureToArray(&g_neighborsData, _neighborsDataArray, &neighborsTextureDesc);
	//cudaMalloc((void**)&_d_agents, sizeof(Agent)*agentsCount);
	//cudaMemcpy(_d_agents, tempAgents, sizeof(Agent)*agentsCount, cudaMemcpyHostToDevice);

	//delete [] tempAgents;
}

__host__  AgentGroup::~AgentGroup(void)
{
	//delete [] _agents;
}

struct CompareAgentHash
{
	__host__ __device__
	bool operator()(int2 a, int2 b)
	{
		return a.y < b.y;
	}
};
__host__ __device__ int GetNodeX(int x, int neighborsDataDim)
{
	return x/neighborsDataDim;
}
__host__ __device__ int GetNodeY(int y, int neighborsDataDim)
{
	return y/neighborsDataDim;
}

__host__ __device__ int GetNodeId( int x, int y, int neighborsDataDim )
{
	int xf = GetNodeX(x, neighborsDataDim);
	int yf = GetNodeY(y, neighborsDataDim);
	int result = xf+yf*neighborsDataDim;
	return result;
}
struct ComputeAgentHash
{
	//Agent* _agents;
	int _neighborsDataDim;
	ComputeAgentHash(/*Agent* agents, */int neighborsDataDim)
	{
		//_agents = agents;
		_neighborsDataDim = neighborsDataDim;
	}
	__device__
	int2 operator()(const int2& item)
	{
		float4 agent = tex1D(g_previousAgentsPositions, /*_agents[*/item.x);
		return make_int2(item.x, GetNodeId(agent.x, agent.y, _neighborsDataDim));
	}
};
struct ComputeNeighborsData
{
	int2* _agentsHashes;
	int _agentsCount;
	ComputeNeighborsData(int2* agentsHashes, int agentCount)
	{
		_agentsHashes = agentsHashes;
		_agentsCount = agentCount;
	}
	__host__ __device__
	int4 operator()(const int4& item)
	{
		int agentHashCounter = 0;
		while(agentHashCounter<_agentsCount && _agentsHashes[agentHashCounter].y!=item.x)
		{
			agentHashCounter++;
		}
		int4 result = make_int4(item.x, agentHashCounter, 0, 0);
		while(agentHashCounter<_agentsCount && _agentsHashes[agentHashCounter++].y==item.x)
		{
			result.z++;
		}
		return result;
	}
};

__global__ void UpdateAgentsKernel(//int3* neighborsData,
	int neighborsDataCount,
	int neighborsDataDim,
	/*int2* agentsHashes,*/
	int agentHashesCount,
	Agent* agents,
	//Agent* prevAgents,
	float4* agentsPositionVelocity,
	float4 boundary,
	float elapsedTime)
{
	int agentIndex = blockIdx.x*blockDim.x + threadIdx.x;
	Agent agent = agents[agentIndex];
		int nodeId = GetNodeId(agent.Position().x, agent.Position().y, neighborsDataDim);
		int neighborsCount = 0;
		int neighborsDataIndices[] = {
			nodeId-neighborsDataDim-1,
			nodeId-neighborsDataDim,
			nodeId-neighborsDataDim+1,
			nodeId- 1,
			nodeId + 0,
			nodeId + 1,
			nodeId + neighborsDataDim-1,
			nodeId + neighborsDataDim,
			nodeId + neighborsDataDim+1,
		};
		for (int i=0;i<9;i++)
		{
			int index = neighborsDataIndices[i];
			if(index < 0 || index >= neighborsDataCount)
				continue;
			neighborsCount += tex1D(g_neighborsData, index).z;
		}
		int* neighbors = new int[neighborsCount];
		int neighborsCounter = 0;
		for (int j = 0;j<9;j++)
		{
			int index = neighborsDataIndices[j];
			if(index<0||index>=neighborsDataCount)
				continue;
			int4 nodeData = tex1D(g_neighborsData, index);
			for (int i = nodeData.y; i<nodeData.y+nodeData.z;i++)
			{
				neighbors[neighborsCounter++] =  tex1D(g_agentsHashes, i).x;
			}
		}
		agent.Update(elapsedTime, neighbors, neighborsCount, boundary);
		delete [] neighbors;
		//syncthreads();
		agents[agentIndex] = agent;
		agentsPositionVelocity[agentIndex] = make_float4(agent.Position().x, agent.Position().y, agent.Velocity().x, agent.Velocity().y);
}

__global__ void GetAgentsPositionVelocity(float4* out)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	out[index] = tex1D(g_previousAgentsPositions, index);
}

__host__ void AgentGroup::Update( float elapsedTime )
{
	thrust::transform(_agentsHashes.begin(),_agentsHashes.end(), _agentsHashes.begin(), 
		ComputeAgentHash(sqrt((float)_neighborsData.size())));
	cudaError_t status = cudaDeviceSynchronize();
	thrust::sort(_agentsHashes.begin(),_agentsHashes.end(), CompareAgentHash());
	status = cudaDeviceSynchronize();
	status = cudaMemcpyToArray(_agentsHashesArray, 0, 0, thrust::raw_pointer_cast(&_agentsHashes[0]), sizeof(int2)*_agentsCount, cudaMemcpyDeviceToDevice);
	thrust::transform(_neighborsData.begin(), _neighborsData.end(), _neighborsData.begin(), ComputeNeighborsData(
		(int2*)thrust::raw_pointer_cast(&_agentsHashes[0]), _agentsCount));
	status = cudaDeviceSynchronize();
	status = cudaMemcpyToArray(_neighborsDataArray, 0, 0, thrust::raw_pointer_cast(&_neighborsData[0]), sizeof(int4)*_neighborsData.size(), cudaMemcpyDeviceToDevice);

	/*int3* neighborsData,
	int neighborsDataCount,
	int neighborsDataDim,
	int2* agentsHashes,
	int agentHashesCount,
	Agent* agents,
	Agent* prevAgents,
	float elapsedTime*/
	UpdateAgentsKernel<<<_agentsCount/128, 128>>>(
//		thrust::raw_pointer_cast(&_neighborsData[0]),
		_neighborsData.size(),
		sqrt(static_cast<float>(_neighborsData.size())),
		//thrust::raw_pointer_cast(&_agentsHashes[0]),
		_agentsCount,
		_d_agents,
		//d_prevAgents,
		_agentPositionVelocity,
		make_float4(0,0,Map::WorldWidth(),Map::WorldHeight()),
		elapsedTime);
	status = cudaDeviceSynchronize();
	//status = cudaFree(d_prevAgents);
	//status = cudaFreeArray(previousAgentsArray);
	//size_t agentSize = sizeof(Agent);
	status = cudaMemcpyToArray(_previousAgentsArray, 0, 0, _agentPositionVelocity, sizeof(float4)*_agentsCount, cudaMemcpyDeviceToDevice);
	
	//status = cudaBindTextureToArray(g_previousAgentsPositions, _previousAgentsArray);
	status = cudaMemcpy(_agents, _d_agents, sizeof(Agent)*_agentsCount, cudaMemcpyDeviceToHost);

	//float4* d_posVelCheck = NULL;
	//status = cudaMalloc(&d_posVelCheck, sizeof(float4)*_agentsCount);
	//GetAgentsPositionVelocity<<<_agentsCount/1024,1024>>>(d_posVelCheck);
	//float4* posVelCheck = static_cast<float4*>(malloc(sizeof(float4)*_agentsCount));
	//status = cudaMemcpy(posVelCheck, d_posVelCheck, sizeof(float4)*_agentsCount, cudaMemcpyDeviceToHost);
}