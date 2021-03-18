#include "Render.h"
#include "Models.h"

#include "VertexPositionColor.h"
#include "ConstantBuffer.h"

#include "DXUT.h"

namespace Render
{
    ID3D11InputLayout* g_pVertexLayout = nullptr;
    ID3D11Buffer* g_pConstantBuffer = NULL;
    CONSTANT_BUFFER cBuffer;

    ID3D11VertexShader* g_pPlainVertexShader = nullptr;
    ID3D11VertexShader* g_pWorldVewProjectionVs = nullptr;
    ID3D11GeometryShader* g_pGeometryShader = nullptr;
    ID3D11PixelShader* g_pPixelShader = nullptr;

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

		V_RETURN(pd3dDevice->CreateVertexShader(PlainVs::g_main, sizeof(PlainVs::g_main), nullptr, &g_pPlainVertexShader));
		V_RETURN(pd3dDevice->CreateVertexShader(WorldVewProjectionVs::g_main, sizeof(WorldVewProjectionVs::g_main), nullptr, &g_pWorldVewProjectionVs));
		V_RETURN(pd3dDevice->CreateGeometryShader(GeometryShader::g_main, sizeof(GeometryShader::g_main), nullptr, &g_pGeometryShader));
		V_RETURN(pd3dDevice->CreatePixelShader(PixelShader::g_main, sizeof(PixelShader::g_main), nullptr, &g_pPixelShader));

		V_RETURN(pd3dDevice->CreateInputLayout(
			VertexPositionColor::VertexDescription,
			VertexPositionColor::VertexDescriptionElementsCount,
			PlainVs::g_main,
			sizeof(PlainVs::g_main),
			&g_pVertexLayout)
		);

		//V_RETURN(pd3dDevice->CreateInputLayout(
		//	VertexPositionColor::VertexDescription,
		//	VertexPositionColor::VertexDescriptionElementsCount,
		//	WorldVewProjectionVs::g_main,
		//	sizeof(WorldVewProjectionVs::g_main),
		//	&g_pVertexPositionColorLayout)
		//);

		// Fill in the subresource data.
		D3D11_SUBRESOURCE_DATA InitData2;
		ZeroMemory(&InitData2, sizeof(D3D11_SUBRESOURCE_DATA));
		InitData2.pSysMem = &cBuffer;
		InitData2.SysMemPitch = 0;
		InitData2.SysMemSlicePitch = 0;

		D3D11_BUFFER_DESC cbDesc;
		ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
		cbDesc.Usage = D3D11_USAGE_DEFAULT;
		cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		cbDesc.ByteWidth = sizeof(CONSTANT_BUFFER);

		// Create the buffer.
		V_RETURN(pd3dDevice->CreateBuffer(&cbDesc, &InitData2, &g_pConstantBuffer));
		
		pd3dContext->PSSetShader(g_pPixelShader, nullptr, 0);

		pd3dContext->GSSetConstantBuffers(0, 1, &g_pConstantBuffer);
		pd3dContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);

		return hr;
    }

    void Release()
    {
		SAFE_RELEASE(g_pVertexLayout);
		SAFE_RELEASE(g_pConstantBuffer)
    }

	void SetViewProjection(CXMMATRIX view, FXMMATRIX projection)
	{
		XMStoreFloat4x4(
			&cBuffer.viewProjection,
			XMMatrixMultiply(
				XMMatrixIdentity(),
				XMMatrixMultiply(
					view,
					projection))
		);
	}

	void Clear(ID3D11DeviceContext* pd3dImmediateContext)
	{
		pd3dImmediateContext->IASetInputLayout(g_pVertexLayout);
		pd3dImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cBuffer, 0, 0);

		float ClearColor[4] = { 0.1f, 0.1f, 0.1f, 0.1f }; // red, green, blue, alpha
		ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
		pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);

		ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
		pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0, 0);
	}

	void RenderTerrain(ID3D11DeviceContext* pd3dImmediateContext)
	{
		UINT stride = sizeof(VertexPositionColor);
		UINT offset = 0;
		pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);

		pd3dImmediateContext->VSSetShader(g_pWorldVewProjectionVs, nullptr, 0);
		pd3dImmediateContext->GSSetShader(nullptr, nullptr, 0);

		pd3dImmediateContext->IASetVertexBuffers(0, 1, &Models::g_pMapVertexBuffer, &stride, &offset);

		pd3dImmediateContext->Draw(Models::terrainVerticesCount, 0);

	}

    void RenderAgents(ID3D11DeviceContext* pd3dImmediateContext)
    {
		UINT stride = sizeof(VertexPositionColor);
		UINT offset = 0;
		pd3dImmediateContext->IASetVertexBuffers(0, 1, &Models::g_pAgentsInstanceData, &stride, &offset);
		pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

		pd3dImmediateContext->VSSetShader(g_pPlainVertexShader, nullptr, 0);
		pd3dImmediateContext->GSSetShader(g_pGeometryShader, nullptr, 0);

		pd3dImmediateContext->Draw(Models::agentsCount, 0);
    }
}