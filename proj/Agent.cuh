#ifndef _AGENT_CUH_
#define _AGENT_CUH_
#include <vector_types.h>
#include <vector_functions.h>

class /*__align__()*/ Agent
{
private:


public:
	float2 target;
	float maxSpeed;
	float3 color;
	//float Id2;
	//float Id3;
	//float Id4;
	//float Id5;
	//float Id6;
	//float Id7;
	//float Id8;
	float2 _position;
	float2 _velocity;
	__host__ __device__ Agent();
	__host__ __device__ Agent(const Agent& object);
	__host__ __device__ ~Agent(void);
	__host__ __device__ float2 Position() const;
	__host__ __device__ void Position(float2 val);
	__host__ __device__ float2 Velocity() const;
	__host__ __device__ void Velocity(float2 val);

	__device__ void Update(float elapsedTime, int* neighbors, int neighborsCount, float4 boundary);
	__device__ float2 WanderSteerUpdate( float wanderDistance, float wanderRadius, float wanderAngle ) const;
	__device__ float2 UnalignedAvoidanceSteerUpdate( int* neighborsPositions, int neighborsCount );
	__device__ float2 SeekSteerUpdate();


	__device__ __host__ float VelocityAngle();
};
#endif