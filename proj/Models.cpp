#include "Models.h"

#include "VertexPositionColor.h"

#include "DXUT.h"

#include <DirectXMath.h>

using namespace DirectX;

namespace Models
{
	namespace Lines
	{
		ID3D11Buffer* gPositions = nullptr;
		ID3D11Buffer* gColors = nullptr;
	}
	namespace Agents
	{
		ID3D11Buffer* gPositions = nullptr;
		ID3D11Buffer* gColors = nullptr;
	}

	uint32_t agentsCount = 0u;
	uint32_t terrainVerticesCount = 0u;

	namespace WorldVewProjectionVs
	{
#include "WorldVewProjectionVs.h"
	}

	namespace PlainVs
	{
#include "PlainVs.h"
	}

	namespace GeometryShader
	{
#include "GeometryShader.h"
	}

	namespace PixelShader
	{
#include "PixelShader.h"
	}

	void GenerateTerrain(
		uint32_t heightNodesCount,
		uint32_t widthNodesCount,
		float worldWidth,
		float worldHeight,
		DirectX::XMFLOAT3** pPositions,
		uint32_t* pVerticesCount
	)
	{
		int verticesCount = (heightNodesCount + 1) * 2 + (widthNodesCount + 1) * 2;
		*pVerticesCount = verticesCount;
		DirectX::XMFLOAT3* positions = new DirectX::XMFLOAT3[verticesCount];

		uint32_t i = 0;

		positions[i++] = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
		positions[i++] = XMFLOAT3{ worldWidth, 0.0f, 0.0f };

		for (uint32_t heightCounter = 1; heightCounter < heightNodesCount + 1; heightCounter++)
		{
			positions[i++] = XMFLOAT3{ 0.0f, worldHeight / heightNodesCount * heightCounter, 0.0f };
			positions[i++] = XMFLOAT3{ worldWidth, worldHeight / heightNodesCount * heightCounter, 0.0f };
		}

		positions[i++] = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
		positions[i++] = XMFLOAT3{ 0.0f, worldHeight, 0.0f };

		for (uint32_t widthCounter = 1; widthCounter < widthNodesCount + 1; widthCounter++)
		{
			positions[i++] = XMFLOAT3{ worldWidth / widthNodesCount * widthCounter, 0.0f, 0.0f };
			positions[i++] = XMFLOAT3{ worldWidth / widthNodesCount * widthCounter, worldHeight, 0.0f };
		}

		*pPositions = positions;
	}

	HRESULT InitTerrain(
		ID3D11Device* pd3dDevice,
		uint32_t heightNodesCount,
		uint32_t widthNodesCount,
		float worldWidth,
		float worldHeight)
	{
		HRESULT hr = S_OK;

		XMFLOAT3* terrainGeometry = nullptr;
		GenerateTerrain(
			heightNodesCount,
			widthNodesCount,
			worldWidth,
			worldHeight,
			&terrainGeometry,
			&terrainVerticesCount);

		D3D11_BUFFER_DESC bufferDesc;
		D3D11_SUBRESOURCE_DATA InitData;

		{
			bufferDesc.ByteWidth = terrainVerticesCount * sizeof(XMFLOAT3);
			bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
			bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bufferDesc.CPUAccessFlags = 0;
			bufferDesc.MiscFlags = 0;
			InitData.pSysMem = terrainGeometry;
			V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, &InitData, &Lines::gPositions));
		}

		{
			XMFLOAT4* colors = new XMFLOAT4[terrainVerticesCount];
			//memset(colors, 0.2f, sizeof(terrainVerticesCount) * agentsCount);
			std::fill(colors, colors + terrainVerticesCount, XMFLOAT4{ 0.2f, 0.2f, 0.2f, 0.1f });
			//for (int i = 0; i < terrainVerticesCount; i++)
			//{
			//	colors[i] = XMFLOAT4{ 0.2f, 0.2f, 0.2f, 0.1f };
			//}
			bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
			bufferDesc.CPUAccessFlags = 0;
			bufferDesc.ByteWidth = sizeof(XMFLOAT4) * terrainVerticesCount;
			InitData.pSysMem = colors;
			V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, &InitData, &Lines::gColors));
		}

		delete[] terrainGeometry;

		return hr;
	}

	HRESULT InitAgents(
		ID3D11Device* pd3dDevice, 
		uint32_t pAgentsCount, 
		XMFLOAT4* colors,
		size_t sizeOfColors,
		size_t sizeOfPositions
	)
	{
		HRESULT hr = S_OK;

		agentsCount = pAgentsCount;

		D3D11_BUFFER_DESC bufferDesc;

		{
			bufferDesc.ByteWidth = sizeOfPositions;
			bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
			bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bufferDesc.MiscFlags = 0;
			V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, nullptr, &Agents::gPositions));
		}

		{
			bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
			bufferDesc.CPUAccessFlags = 0;
			bufferDesc.ByteWidth = sizeOfColors;
			D3D11_SUBRESOURCE_DATA InitData;
			InitData.pSysMem = colors;
			V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, &InitData, &Agents::gColors));
		}

		return hr;
	}

	void Release()
	{
		SAFE_RELEASE(Lines::gColors);
		SAFE_RELEASE(Lines::gPositions);

		SAFE_RELEASE(Agents::gColors);
		SAFE_RELEASE(Agents::gPositions);
	}

	void UpdateAgents(ID3D11DeviceContext* pd3dContext, XMFLOAT2* positions, size_t sizeOfPositions)
	{
		D3D11_MAPPED_SUBRESOURCE mappedSubresource;
		ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));

		HRESULT res = pd3dContext->Map(Agents::gPositions, 0, D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubresource);
		memcpy(mappedSubresource.pData, positions, sizeOfPositions);
		pd3dContext->Unmap(Agents::gPositions, 0);
	}
}