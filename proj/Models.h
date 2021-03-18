#pragma once

#include "VertexPositionColor.h"

#include <d3d11.h>

namespace Models
{
    extern ID3D11Buffer* g_pMapVertexBuffer;
    extern ID3D11Buffer* g_pAgentsInstanceData;

	extern uint32_t agentsCount;
	extern uint32_t terrainVerticesCount;

	HRESULT Init(
		ID3D11Device* pd3dDevice,
		uint32_t agentsCount,
		uint32_t heightNodesCount,
		uint32_t widthNodesCount,
		float worldWidth,
		float worldHeight);
    void Release();

	void UpdateAgents(ID3D11DeviceContext* pd3dContext, VertexPositionColor* agentsData);
	HRESULT RenderTerrain(ID3D11DeviceContext* pd3dImmediateContext);

}