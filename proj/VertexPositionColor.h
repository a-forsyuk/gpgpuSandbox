#pragma once

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

const D3D11_INPUT_ELEMENT_DESC VertexPositionColor::VertexDescription[] =
{
	{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
};

const int VertexPositionColor::VertexDescriptionElementsCount = 2;
