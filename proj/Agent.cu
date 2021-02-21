#include "AgentsKernel.cuh"
#include "CustomVectorOperations.cuh"
#include <math_constants.h>
#include <curand_kernel.h>
__host__ __device__ Agent::Agent()
{
	_velocity = make_float2(0,0);
	target = make_float2(0,0);
	color = make_float3(0,0,0);
}

__host__ __device__ Agent::Agent( const Agent& object )
{
	_velocity = object._velocity;
	_position = object._position;
	target = object.target;
	color = object.color;
}


__host__ __device__ Agent::~Agent(void)
{
}

__host__ __device__ float2 Agent::Position() const
{
	return _position;
}

__host__ __device__ void Agent::Position( float2 val )
{
	_position = val;
}

__host__ __device__ float2 Agent::Velocity() const
{
	return _velocity;
}

__host__ __device__ void Agent::Velocity( float2 val )
{
	_velocity = val;
}

__device__ void Agent::Update( float elapsedTime, int* neighbors, int neighborsCount, float4 boundary )
{
	maxSpeed = elapsedTime*5.0f;
	//curandState_t randState;
	//curand_init (5, 50, 0, &randState);
	float2 steeringForce = SeekSteerUpdate();//WanderSteerUpdate(100.0f, 5.0f, curand(&randState)*1.0f - 1.0f * 0.5f);
	_velocity += steeringForce;
	Truncate(&_velocity, maxSpeed);
	steeringForce += UnalignedAvoidanceSteerUpdate(neighbors, neighborsCount);


	float2 newPosition = _position + _velocity;
	if(newPosition.x>boundary.x&&
		newPosition.y>boundary.y&&
		newPosition.x<boundary.z&&
		newPosition.y<boundary.w)
		_position = newPosition;
	else
		_velocity = Zero<float2>()-_velocity;
}

__device__ float2 Agent::WanderSteerUpdate( float wanderDistance, float wanderRadius, float wanderAngle ) const
{
	float2 wanderCenter = float2(_velocity);
	Normalize(&wanderCenter);
	Multiply(&wanderCenter, wanderDistance);
	float2 circlePoint = Create<float2>(wanderRadius, wanderAngle);
	/*_wanderAngle += rand()*wanderAngleRange - _wanderAngleRange * 0.5f;*/
	float2 force = wanderCenter + circlePoint;
	Normalize(&force);
	return force;
}

__device__ float2 Agent::UnalignedAvoidanceSteerUpdate( int* neighborsPositions, int neighborsCount )
{
	float2 totalForce = make_float2(0.0f,0.0f);
	float localMaxSpeed = maxSpeed*5;
	for (int i = 0 ; i < neighborsCount ; i++)
	{
		float4 otherAgentPosVel = tex1D(g_previousAgentsPositions,neighborsPositions[i]);
		float2 agentPosition = make_float2(otherAgentPosVel.x, otherAgentPosVel.y);
		float2 agentVelocity = make_float2(otherAgentPosVel.z, otherAgentPosVel.w);
		float2 forward = _velocity;
		Normalize(&forward);
		//Agent otherAgent = neighborsPositions[i];
		float2 distance = agentPosition - _position;
		if(LengthSquared(distance)<0.00001f)
			continue;
		Normalize(&agentVelocity);
		Multiply(&agentVelocity, 15.0f);
		float2 diff = agentPosition + agentVelocity - _position;
		float dotProduct = DotProduct(diff, forward);
		if(dotProduct > 0)
		{
			float2 ray = forward;
			Multiply(&ray, 15.0f);
			float2 projection = forward;
			Multiply(&projection, dotProduct);
			float dist = GetLength(projection - diff);
			if(dist<7.0f  && GetLength(projection)<GetLength(ray))
			{
				float2 force = forward;
				Multiply(&force, localMaxSpeed);
				SetAngle(&force, GetAngle(force)+CUDART_PIO4_F/4.0f*Sign(diff, _velocity));
				Multiply(&force, 1-GetLength(projection)/GetLength(ray));
				//totalForce += force;
				_velocity += force;
				Multiply(&_velocity, GetLength(projection)/GetLength(ray));
			}
		}
	}
	return totalForce;
}

__device__ __host__ float Agent::VelocityAngle()
{
	return GetAngle(_velocity);
}

__device__ float2 Agent::SeekSteerUpdate()
{
	float2 desiredVelocity = target - _position;
	Normalize(&desiredVelocity);
	Multiply(&desiredVelocity, 5.0f);
	float2 force = desiredVelocity - _velocity;
	return force;
}
