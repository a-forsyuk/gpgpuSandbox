#pragma once

#include <DirectXMath.h>
#include <d3d11.h>
#include <stdint.h>

namespace Models
{
	namespace Lines
	{
		extern ID3D11Buffer* gPositions;
		extern ID3D11Buffer* gColors;
	}
	namespace Agents
	{
		extern ID3D11Buffer* gPositions;
		extern ID3D11Buffer* gColors;
	}

	extern uint32_t agentsCount;
	extern uint32_t terrainVerticesCount;

	HRESULT InitTerrain(ID3D11Device* pd3dDevice, 
		uint32_t heightNodesCount, uint32_t widthNodesCount,
		float worldWidth,float worldHeight);
	HRESULT InitAgents(ID3D11Device* pd3dDevice, 
		uint32_t pAgentsCount, 
		DirectX::XMFLOAT4* colors, size_t sizeOfColors, 
		size_t sizeOfPositions);
    void Release();

	void UpdateAgents(ID3D11DeviceContext* pd3dContext, DirectX::XMFLOAT2* positions, size_t sizeOfPositions);
	HRESULT RenderTerrain(ID3D11DeviceContext* pd3dImmediateContext);

}