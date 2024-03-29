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
	if ((frontIndex % 2) == 0)
	{
		return frontBuffer;
	}
	return backBuffer;
}

template<typename T>
T* DoubleBuffer<T>::GetBack() const
{
	if ((frontIndex % 2) == 1)
	{
		return frontBuffer;
	}
	return backBuffer;
}

template<typename T>
T** DoubleBuffer<T>::GetFrontPtr()
{
	if ((frontIndex % 2) == 0)
	{
		return &frontBuffer;
	}
	return &backBuffer;
}

template<typename T>
T** DoubleBuffer<T>::GetBackPtr()
{
	if ((frontIndex % 2) == 1)
	{
		return &frontBuffer;
	}
	return &backBuffer;
}

template<typename T>
void DoubleBuffer<T>::Swap()
{
	frontIndex++;
	//std::swap(frontBuffer, backBuffer);
}

template<typename T>
void DoubleBuffer<T>::Release()
{
	frontBuffer->Release();
	frontBuffer = nullptr;

	backBuffer->Release();
	backBuffer = nullptr;
}