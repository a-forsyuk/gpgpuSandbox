#include "Models.h"

#include "VertexPositionColor.h"

#include "DXUT.h"

namespace Models
{
    ID3D11Buffer* g_pMapVertexBuffer = nullptr;
    ID3D11Buffer* g_pAgentsInstanceData = nullptr;

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
		VertexPositionColor** data,
		uint32_t* pVerticesCount
	)
	{
		int verticesCount = (heightNodesCount + 1) * 2 + (widthNodesCount + 1) * 2;
		*pVerticesCount = verticesCount;
		VertexPositionColor* mapLinesVertices = new VertexPositionColor[verticesCount];

		for (int i = 0; i < verticesCount; i++)
		{
			mapLinesVertices[i].color = XMFLOAT4{ 0.2f,0.2f,0.2f,0.1f };
		}

		int i = 0;

		mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
		mapLinesVertices[i++].position = XMFLOAT3{ worldWidth, 0.0f, 0.0f };

		for (int heightCounter = 1; heightCounter < heightNodesCount + 1; heightCounter++)
		{
			mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, worldHeight / heightNodesCount * heightCounter, 0.0f };
			mapLinesVertices[i++].position = XMFLOAT3{ worldWidth, worldHeight / heightNodesCount * heightCounter, 0.0f };
		}

		mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
		mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, worldHeight, 0.0f };

		for (int widthCounter = 1; widthCounter < widthNodesCount + 1; widthCounter++)
		{
			mapLinesVertices[i++].position = XMFLOAT3{ worldWidth / widthNodesCount * widthCounter, 0.0f, 0.0f };
			mapLinesVertices[i++].position = XMFLOAT3{ worldWidth / widthNodesCount * widthCounter, worldHeight, 0.0f };
		}

		*data = mapLinesVertices;
	}

    HRESULT Init(
		ID3D11Device* pd3dDevice, 
		uint32_t pAgentsCount,
		uint32_t heightNodesCount,
		uint32_t widthNodesCount,
		float worldWidth,
		float worldHeight)
    {
        HRESULT hr = S_OK;

		agentsCount = pAgentsCount;

		D3D11_BUFFER_DESC bufferDesc;
		bufferDesc.ByteWidth = agentsCount * sizeof(VertexPositionColor);
		bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		bufferDesc.MiscFlags = 0;

		V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, NULL, &g_pAgentsInstanceData));

		VertexPositionColor* terrainGeometry = nullptr;
		GenerateTerrain(
			heightNodesCount,
			widthNodesCount,
			worldWidth,
			worldHeight,
			&terrainGeometry,
			&terrainVerticesCount);

		bufferDesc.Usage = D3D11_USAGE_DEFAULT;
		bufferDesc.CPUAccessFlags = 0;
		bufferDesc.ByteWidth = sizeof(VertexPositionColor) * terrainVerticesCount;

		D3D11_SUBRESOURCE_DATA InitData;
		InitData.pSysMem = terrainGeometry;
		V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, &InitData, &g_pMapVertexBuffer));

		delete terrainGeometry;

		return hr;
    }

    void Release()
    {
		SAFE_RELEASE(g_pMapVertexBuffer);
		SAFE_RELEASE(g_pAgentsInstanceData);
    }

	void UpdateAgents(ID3D11DeviceContext* pd3dContext, VertexPositionColor* agentsData)
	{
		D3D11_MAPPED_SUBRESOURCE mappedSubresource;
		ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));

		HRESULT res = pd3dContext->Map(g_pAgentsInstanceData, 0, D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubresource);
		memcpy(mappedSubresource.pData, agentsData, sizeof(VertexPositionColor) * agentsCount);
		pd3dContext->Unmap(g_pAgentsInstanceData, 0);
	}
}