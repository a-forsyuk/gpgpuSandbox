#include "Render.h"
#include "Models.h"

#include "VertexPositionColor.h"
#include "AgentVertex.h"
#include "ConstantBuffer.h"

#include "DXUT.h"

namespace Render
{
    ID3D11InputLayout* vertexLayout = nullptr;
	ID3D11InputLayout* agentVertexLayout = nullptr;
    ID3D11Buffer* constantBuffer = NULL;
    CONSTANT_BUFFER constantBufferData;

    ID3D11VertexShader* plainVertexShader = nullptr;
    ID3D11VertexShader* worldVewProjectionVs = nullptr;
    ID3D11GeometryShader* geometryShader = nullptr;
    ID3D11PixelShader* pixelShader = nullptr;

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

    HRESULT Init(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dContext, uint32_t agentsCount)
    {
        HRESULT hr = S_OK;

		V_RETURN(pd3dDevice->CreateVertexShader(PlainVs::g_main, sizeof(PlainVs::g_main), nullptr, &plainVertexShader));
		V_RETURN(pd3dDevice->CreateVertexShader(WorldVewProjectionVs::g_main, sizeof(WorldVewProjectionVs::g_main), nullptr, &worldVewProjectionVs));
		V_RETURN(pd3dDevice->CreateGeometryShader(GeometryShader::g_main, sizeof(GeometryShader::g_main), nullptr, &geometryShader));
		V_RETURN(pd3dDevice->CreatePixelShader(PixelShader::g_main, sizeof(PixelShader::g_main), nullptr, &pixelShader));

		V_RETURN(pd3dDevice->CreateInputLayout(
			VertexPositionColor::VertexDescription.data(),
			VertexPositionColor::VertexDescription.size(),
			WorldVewProjectionVs::g_main,
			(uint32_t)sizeof(WorldVewProjectionVs::g_main),
			&vertexLayout)
		);

		V_RETURN(pd3dDevice->CreateInputLayout(
			AgentVertex::Description.data(),
			AgentVertex::Description.size(),
			PlainVs::g_main,
			(uint32_t)sizeof(PlainVs::g_main),
			&agentVertexLayout)
		);

		// Fill in the subresource data.
		D3D11_SUBRESOURCE_DATA InitData2;
		ZeroMemory(&InitData2, sizeof(D3D11_SUBRESOURCE_DATA));
		InitData2.pSysMem = &constantBufferData;
		InitData2.SysMemPitch = 0;
		InitData2.SysMemSlicePitch = 0;

		D3D11_BUFFER_DESC cbDesc;
		ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
		cbDesc.Usage = D3D11_USAGE_DEFAULT;
		cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbDesc.ByteWidth = sizeof(CONSTANT_BUFFER);

		// Create the buffer.
		V_RETURN(pd3dDevice->CreateBuffer(&cbDesc, &InitData2, &constantBuffer));
		
		pd3dContext->PSSetShader(pixelShader, nullptr, 0);

		pd3dContext->GSSetConstantBuffers(0, 1, &constantBuffer);
		pd3dContext->VSSetConstantBuffers(0, 1, &constantBuffer);

		return hr;
    }

    void Release()
    {
		SAFE_RELEASE(vertexLayout);
		SAFE_RELEASE(constantBuffer)
    }

	void SetViewProjection(CXMMATRIX view, FXMMATRIX projection)
	{
		XMStoreFloat4x4(
			&constantBufferData.viewProjection,
			XMMatrixMultiply(
				XMMatrixIdentity(),
				XMMatrixMultiply(
					view,
					projection))
		);
	}

	void Clear(ID3D11DeviceContext* pd3dImmediateContext)
	{
		pd3dImmediateContext->UpdateSubresource(constantBuffer, 0, nullptr, &constantBufferData, 0, 0);

		float ClearColor[4] { 0.1f, 0.1f, 0.1f, 0.1f }; // red, green, blue, alpha
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);

		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0, 0);
	}

	void BindTerrain(ID3D11DeviceContext* pd3dImmediateContext)
	{
		uint32_t strides[]{ sizeof(XMFLOAT3), sizeof(XMFLOAT4) };
		uint32_t offsets[]{ 0u, 0u };
		ID3D11Buffer* buffers[]{ Models::Lines::gPositions, Models::Lines::gColors };

		pd3dImmediateContext->IASetInputLayout(vertexLayout);

		pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

		pd3dImmediateContext->VSSetShader(worldVewProjectionVs, nullptr, 0);
		pd3dImmediateContext->GSSetShader(nullptr, nullptr, 0);

		pd3dImmediateContext->IASetVertexBuffers(0, 2, buffers, strides, offsets);
	}

	void UnbindTerrain(ID3D11DeviceContext* pd3dImmediateContext)
	{
		uint32_t strides[]{ 0u, 0u };
		uint32_t offsets[]{ 0u, 0u };
		ID3D11Buffer* buffers[]{ nullptr, nullptr };
		pd3dImmediateContext->IASetVertexBuffers(0, 2, buffers, strides, offsets);
	}

	void RenderTerrain(ID3D11DeviceContext* pd3dImmediateContext)
	{
		BindTerrain(pd3dImmediateContext);

		pd3dImmediateContext->Draw(Models::terrainVerticesCount, 0);

		UnbindTerrain(pd3dImmediateContext);
	}

    void RenderAgents(ID3D11DeviceContext* pd3dImmediateContext)
    {
		uint32_t strides[]{ sizeof(XMFLOAT2), sizeof(XMFLOAT4) };
		uint32_t offsets[]{ 0u, 0u };
		ID3D11Buffer* buffers[]{ Models::Agents::gPositions.GetFront(), Models::Agents::gColors };

		pd3dImmediateContext->IASetInputLayout(agentVertexLayout);

		pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

		pd3dImmediateContext->IASetVertexBuffers(0, 2, buffers, strides, offsets);

		pd3dImmediateContext->VSSetShader(plainVertexShader, nullptr, 0);
		pd3dImmediateContext->GSSetShader(geometryShader, nullptr, 0);

		pd3dImmediateContext->Draw(Models::agentsCount, 0);

		UnbindTerrain(pd3dImmediateContext);

		Models::Agents::gPositions.Swap();
    }
}