#pragma once

#include <d3d11.h>

#include <array>
#include <assert.h>

template<typename T>
class DoubleBuffer
{
public:

	void Init(T* front, T* back);

	T* GetFront() const;
	T* GetBack() const;

	T** GetFrontPtr();
	T** GetBackPtr();

	void Swap();

	void Release();

private:
	T* frontBuffer;
	T* backBuffer;

	u_char frontIndex = 0;
};

template<typename T>
void DoubleBuffer<T>::Init(T* front, T* back)
{
	assert(front != nullptr);
	assert(back != nullptr);

	frontBuffer = front;
	backBuffer = back;
}

template<typename T>
T* DoubleBuffer<T>::GetFront() const
{
	return frontBuffer;
}

template<typename T>
T* DoubleBuffer<T>::GetBack() const
{
	return backBuffer;
}

template<typename T>
T** DoubleBuffer<T>::GetFrontPtr()
{
	return &frontBuffer;
}

template<typename T>
T** DoubleBuffer<T>::GetBackPtr()
{
	return &backBuffer;
}

template<typename T>
void DoubleBuffer<T>::Swap()
{
	std::swap(frontBuffer, backBuffer);
}

template<typename T>
void DoubleBuffer<T>::Release()
{
	frontBuffer->Release();
	frontBuffer = nullptr;

	backBuffer->Release();
	backBuffer = nullptr;
}