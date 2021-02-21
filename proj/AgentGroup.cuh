#ifndef _AGENT_GROUP_CUH_
#define _AGENT_GROUP_CUH_
#include "Agent.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
class AgentGroup
{
private:
	Agent* _d_agents;
	Agent* _agents;
	cudaArray* _previousAgentsArray;
	cudaArray* _neighborsDataArray;
	cudaArray* _agentsHashesArray;
	float4* _agentPositionVelocity;
	int _agentsCount;

	thrust::device_vector<int2> _agentsHashes;
	thrust::device_vector<int4> _neighborsData;
public:
	__host__ __device__ Agent* Agents() const { return (Agent*)thrust::raw_pointer_cast(&_agents[0]); }
	__host__ __device__ int AgentsCount() const { return _agentsCount; }

	__host__  void Update(float elapsedTime);

	__host__  AgentGroup(int agentsCount);
	__host__  ~AgentGroup(void);
};
#endif