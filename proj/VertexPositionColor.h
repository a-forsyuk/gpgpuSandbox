#pragma once

#include <d3d11.h>
#include <DirectXMath.h>

using namespace DirectX;

struct VertexPositionColor
{
public:
	XMFLOAT3 position;
	XMFLOAT4 color;

	static const D3D11_INPUT_ELEMENT_DESC VertexDescription[];
	static const int VertexDescriptionElementsCount;
};
