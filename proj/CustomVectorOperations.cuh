#pragma once
#ifndef _CUSTOM_VECTOR_OPERATIONS_CUH_
#define _CUSTOM_VECTOR_OPERATIONS_CUH_

///Declarations

template<typename T> __host__ __device__ T Zero();
template<typename T> __host__ __device__ void Normalize(T* vector);
template<typename T> __host__ __device__ float LengthSquared(T vector);
template<typename T> __host__ __device__ float GetLength(T vector);
template<typename T> __host__ __device__ void SetLength(T* vector, float length);
template<typename T> __host__ __device__ void Multiply(T* vector, float scalar);
template<typename T> __host__ __device__ T Create(float length, float angle);
template<typename T> __host__ __device__ void Truncate(T* vector, float length);
template<typename T> __host__ __device__ float DotProduct(T vector1, T vector2);
template<typename T> __host__ __device__ float GetAngle(T vector);
template<typename T> __host__ __device__ void SetAngle(T* vector, float angle);
template<typename T> __host__ __device__ T Perp(T vector);
template<typename T> __host__ __device__ int Sign(T vector1, T vector2);

//Implementations

template<typename T> __host__ __device__ 
T Zero()
{
	T result;
	result.x = 0;
	result.y = 0;
	return result;
}

template<typename T> __host__ __device__ 
void Normalize(T* vector)
{
	float length = GetLength(*vector);
	if(length<=0.0001f)
	{
		vector->x = 1;
	}
	else
	{
		vector->x /= length;
		vector->y /= length;
	}
}

template<typename T> __host__ __device__ 
float LengthSquared(T vector)
{
	return vector.x*vector.x + vector.y*vector.y;
}

template<typename T> __host__ __device__ 
float GetLength(T vector)
{
	return sqrt(LengthSquared(vector));
}

template<typename T> __host__ __device__ 
void SetLength(T* vector, float length)
{
	float angle = GetAngle(*vector);
	vector->x = cos(angle) * length;
	vector->y = sin(angle) * length;
}

template<typename T>
T Create(float length, float angle)
{
	T result;
	result.x = cos(angle)*length;
	result.y = sin(angle)*length;
	return result;
}

template<typename T> __host__ __device__ 
void Multiply(T* vector, float scalar)
{
	vector->x *= scalar;
	vector->y *= scalar;
}

template<typename T> __host__ __device__ 
void Truncate(T* vector, float length)
{
	float actualLength = GetLength(*vector);
	if(length<actualLength)
	{
		SetLength(vector, length);
	}
}

template<typename T> __host__ __device__ 
float DotProduct(T vector1, T vector2)
{
	return vector1.x*vector2.x+vector1.y*vector2.y;
}

template<typename T> __host__ __device__ 
float GetAngle(T vector)
{
	return atan2(vector.y, vector.x);
}
template<typename T> __host__ __device__ 
void SetAngle(T* vector, float angle)
{
	float length = GetLength(*vector);
	vector->x = cos(angle) * length;
	vector->y = sin(angle) * length;
}

template<typename T> __host__ __device__ 
	T Perp(T vector)
{
	T result;
	result.x = -vector.y;
	result.y = vector.x;
	return result;
}

template<typename T> __host__ __device__ 
	int Sign(T vector1, T vector2)
{
	return DotProduct(Perp(vector1), vector2) < 0 ? -1 : 1;
}

__host__ __device__ float2 operator + (float2& lhs, const float2 &rhs) {
	return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

__host__ __device__ float2& operator += (float2& lhs, const float2 &rhs) {
	lhs.x += rhs.x;
	lhs.y += rhs.y;
	return lhs;
}

__host__ __device__ float2 operator - (float2& lhs, const float2 &rhs) {
	return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

__host__ __device__ float2& operator -= (float2& lhs, const float2 &rhs) {
	lhs.x -= rhs.x;
	lhs.y -= rhs.y;
	return lhs;
}
#endif