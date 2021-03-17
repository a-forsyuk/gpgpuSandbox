// CUDA_first.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CUDA_first.h"
#include <assert.h>
#include <iostream>
#include "DXUT.h"
#include "DXUTcamera.h"

#include "VertexPositionColor.h"
#include "TransformColorInstBatch.h"
#include "ConstantBuffer.h"

#include "SDKmisc.h"

#include "AgentsKernel.cuh"

#include <DirectXMath.h>

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

using namespace DirectX;

//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
ID3D11InputLayout* g_pVertexLayout = nullptr;
ID3D11InputLayout* g_pVertexPositionColorLayout = nullptr;
ID3D11Buffer* g_pAxisVertexBuffer = nullptr;
ID3D11Buffer* g_pMapVertexBuffer = nullptr;
ID3D11Buffer* g_pAgentsInstanceData = nullptr;
ID3D11Buffer* g_pConstantBuffer = NULL;
CONSTANT_BUFFER cBuffer;

ID3D11VertexShader* g_pPlainVertexShader = nullptr;
ID3D11VertexShader* g_pWorldVewProjectionVs = nullptr;
ID3D11GeometryShader* g_pGeometryShader = nullptr;
ID3D11PixelShader* g_pPixelShader = nullptr;

CModelViewerCamera g_Camera;
const UINT AgentsCount = 6144;
VertexPositionColor* instanceData = nullptr;

uint32_t widthNodesCount = 0;
uint32_t heightNodesCount = 0;

//--------------------------------------------------------------------------------------
// Reject any D3D10 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable(_In_ const CD3D11EnumAdapterInfo* AdapterInfo, _In_ UINT Output, _In_ const CD3D11EnumDeviceInfo* DeviceInfo,
	_In_ DXGI_FORMAT BackBufferFormat, _In_ bool bWindowed, _In_opt_ void* pUserContext)
{

	return DeviceInfo->DeviceType == D3D_DRIVER_TYPE_HARDWARE && bWindowed;
}

//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBufferSurfaceDesc,
	void* pUserContext )
{
	HRESULT hr = S_OK;

	V_RETURN(pd3dDevice->CreateVertexShader(PlainVs::g_main, sizeof(PlainVs::g_main), nullptr, &g_pPlainVertexShader));
	V_RETURN(pd3dDevice->CreateVertexShader(WorldVewProjectionVs::g_main, sizeof(WorldVewProjectionVs::g_main), nullptr, &g_pWorldVewProjectionVs));
	V_RETURN(pd3dDevice->CreateGeometryShader(GeometryShader::g_main, sizeof(GeometryShader::g_main), nullptr, &g_pGeometryShader));
	V_RETURN(pd3dDevice->CreatePixelShader(PixelShader::g_main, sizeof(PixelShader::g_main), nullptr, &g_pPixelShader));

	V_RETURN( pd3dDevice->CreateInputLayout( 
		VertexPositionColor::VertexDescription, 
		VertexPositionColor::VertexDescriptionElementsCount, 
		PlainVs::g_main,
		sizeof(PlainVs::g_main),
		&g_pVertexLayout ) 
	);

	V_RETURN( pd3dDevice->CreateInputLayout(
		VertexPositionColor::VertexDescription, 
		VertexPositionColor::VertexDescriptionElementsCount, 
		WorldVewProjectionVs::g_main,
		sizeof(WorldVewProjectionVs::g_main),
		&g_pVertexPositionColorLayout ) 
	);

	D3D11_BUFFER_DESC bufferDesc;
	bufferDesc.ByteWidth = AgentsCount * sizeof( VertexPositionColor ),
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC,
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER,
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE,
	bufferDesc.MiscFlags = 0;

	V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, NULL, &g_pAgentsInstanceData ) );
	UINT stride = sizeof(VertexPositionColor);
	UINT offset = 0;
	DXUTGetD3D11DeviceContext()->IASetVertexBuffers(0, 1, &g_pAgentsInstanceData, &stride, &offset);

	///Axis///
	VertexPositionColor axisVertices[] =
	{
		{ XMFLOAT3{ 100.0f, 0.0f, 0.0f }, XMFLOAT4{ 1.0f, 0.0f, 0.0f, 1.0f} },
		{ XMFLOAT3{ 0.0f, 0.0f, 0.0f }, XMFLOAT4{ 1.0f, 0.0f, 0.0f, 1.0f} },

		{ XMFLOAT3{ 0.0f, 100.0f, 0.0f }, XMFLOAT4{ 0.0f, 1.0f, 0.0f, 1.0f} },
		{ XMFLOAT3{ 0.0f, 0.0f, 0.0f }, XMFLOAT4{ 0.0f, 1.0f, 0.0f, 1.0f} },

		{ XMFLOAT3{ 0.0f, 0.0f, 100.0f }, XMFLOAT4{ 0.0f, 0.0f, 1.0f, 1.0f} },
		{ XMFLOAT3{ 0.0f, 0.0f, 0.0f }, XMFLOAT4{ 0.0f, 0.0f, 1.0f, 1.0f} },
	};

	D3D11_BUFFER_DESC bd;
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof( VertexPositionColor ) * 6;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	
	D3D11_SUBRESOURCE_DATA InitData;
	InitData.pSysMem = axisVertices;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, &InitData, &g_pAxisVertexBuffer ) );

	CUDASystems::GetMapNodesDimensions(&widthNodesCount, &heightNodesCount);

	float worldWidth = 0;
	float worldHeight = 0;
	CUDASystems::GetMapDimensions(&worldWidth, &worldHeight);

	///Map///
	int verticesCount = (heightNodesCount + 1) * 2 + (widthNodesCount + 1) * 2;
	VertexPositionColor* mapLinesVertices = new VertexPositionColor[verticesCount];

	for (int i=0;i<verticesCount;i++)
	{
		mapLinesVertices[i].color = XMFLOAT4{ 0.2f,0.2f,0.2f,0.1f };
	}

	int i = 0;

	mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
	mapLinesVertices[i++].position = XMFLOAT3{ worldWidth, 0.0f, 0.0f };

	for (int heightCounter = 1; heightCounter<heightNodesCount + 1; heightCounter++)
	{
		mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, worldHeight / heightNodesCount * heightCounter, 0.0f };
		mapLinesVertices[i++].position = XMFLOAT3{ worldWidth, worldHeight / heightNodesCount * heightCounter, 0.0f };
	}

	mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
	mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, worldHeight, 0.0f };

	for (int widthCounter = 1;widthCounter<widthNodesCount+1;widthCounter++)
	{
		mapLinesVertices[i++].position = XMFLOAT3{ worldWidth / widthNodesCount * widthCounter, 0.0f, 0.0f };
		mapLinesVertices[i++].position = XMFLOAT3{ worldWidth / widthNodesCount * widthCounter, worldHeight, 0.0f };
	}

	bd.ByteWidth = sizeof( VertexPositionColor ) * verticesCount;

	InitData.pSysMem = mapLinesVertices;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, &InitData, &g_pMapVertexBuffer ) );

	// Initialize the view matrix
	XMVECTORF32 Eye { worldWidth / 2.0f, 0.0f, 100.0f };
	XMVECTORF32 At { worldWidth / 2.0f, worldHeight / 2.0f, 0.0f };
	g_Camera.SetViewParams( Eye, At );

	instanceData = new VertexPositionColor[AgentsCount];
	memset(instanceData, 0, sizeof(VertexPositionColor) * AgentsCount);

	float* agentColors = nullptr;
	CUDASystems::MapColors(&agentColors);

	for (uint32_t i = 0; i < AgentsCount; i++)
	{
		memcpy(&instanceData[i].color, agentColors + i * 4u, sizeof(float) * 4u);
	}

	// Fill in the subresource data.
	D3D11_SUBRESOURCE_DATA InitData2;
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

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
// Create and set the depth stencil texture if needed
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain(_In_ ID3D11Device* pd3dDevice, _In_ IDXGISwapChain* pSwapChain, _In_ const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, _In_opt_ void* pUserContext)
{
	// Setup the camera's projection parameters
	float fAspectRatio = static_cast<float>(pBackBufferSurfaceDesc->Width ) /
		static_cast<float>(pBackBufferSurfaceDesc->Height );
	g_Camera.SetProjParams( XM_PI / 4, fAspectRatio, 0.1f, 5000.0f );
	g_Camera.SetWindow(pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
	g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );
	g_Camera.SetEnablePositionMovement(true);

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D10 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext)
{
	float ClearColor[4] = { 0.1f, 0.1f, 0.1f, 0.1f }; // red, green, blue, alpha
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );

	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0, 0 );

	XMStoreFloat4x4(
		&cBuffer.viewProjection,
		XMMatrixMultiply(
			XMMatrixIdentity(),
			XMMatrixMultiply(
				g_Camera.GetViewMatrix(),
				g_Camera.GetProjMatrix()))
	);
	pd3dImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cBuffer, 0, 0);

	// Set vertex buffer
	UINT stride = sizeof( VertexPositionColor );
	UINT offset = 0;
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pAxisVertexBuffer, &stride, &offset );
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
	pd3dImmediateContext->IASetInputLayout( g_pVertexPositionColorLayout );

	pd3dImmediateContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
	pd3dImmediateContext->VSSetShader(g_pWorldVewProjectionVs, nullptr, 0);
	pd3dImmediateContext->GSSetShader(nullptr, nullptr, 0);
	pd3dImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);
	pd3dImmediateContext->Draw(6, 0);

	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pMapVertexBuffer, &stride, &offset );

	pd3dImmediateContext->Draw((heightNodesCount + 1) * 2 + (widthNodesCount + 1) * 2, 0);

	D3D11_MAPPED_SUBRESOURCE mappedSubresource;
	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	HRESULT res = DXUTGetD3D11DeviceContext()->Map(g_pAgentsInstanceData, 0, D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubresource);

	float* agentPositions = nullptr;
	CUDASystems::MapPositions(&agentPositions);
	for (uint32_t i = 0; i < AgentsCount; i++)
	{
		memcpy(&instanceData[i].position, agentPositions + i * 2u, sizeof(float)* 2u);
	}
	memcpy(mappedSubresource.pData, instanceData, sizeof(VertexPositionColor) * AgentsCount);
	DXUTGetD3D11DeviceContext()->Unmap(g_pAgentsInstanceData, 0);
	stride = sizeof(VertexPositionColor);
	offset = 0;
	DXUTGetD3D11DeviceContext()->IASetVertexBuffers(0, 1, &g_pAgentsInstanceData, &stride, &offset);
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
	pd3dImmediateContext->IASetInputLayout( g_pVertexLayout );

	pd3dImmediateContext->VSSetShader(g_pPlainVertexShader, nullptr, 0);
	pd3dImmediateContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
	pd3dImmediateContext->GSSetShader(g_pGeometryShader, nullptr, 0);
	pd3dImmediateContext->GSSetConstantBuffers(0, 1, &g_pConstantBuffer);
	pd3dImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);
	pd3dImmediateContext->Draw(AgentsCount, 0);
}

//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	SAFE_RELEASE( g_pAxisVertexBuffer );
	SAFE_RELEASE( g_pMapVertexBuffer );
	SAFE_RELEASE( g_pAgentsInstanceData);
	SAFE_RELEASE( g_pVertexLayout );
	SAFE_RELEASE( g_pVertexPositionColorLayout );
	SAFE_RELEASE(g_pConstantBuffer)
	CUDASystems::Release();
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
	CUDASystems::Update(fElapsedTime);
	g_Camera.FrameMove(fElapsedTime);
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
	void* pUserContext )
{
	g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

	return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	if( bKeyDown )
	{
		switch( nChar )
		{
		case VK_F1: // Change as needed                
			break;
		}
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	CUDASystems::Init(AgentsCount);

	DXUTSetCallbackD3D11DeviceAcceptable(IsD3D11DeviceAcceptable);
	DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
	DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
	DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
	DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);
	DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender);

	DXUTSetCallbackMsgProc(MsgProc);
	DXUTSetCallbackKeyboard(OnKeyboard);
	DXUTSetCallbackFrameMove(OnFrameMove);
	DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);

	DXUTInit(true, true, NULL); // Parse the command line, show msgboxes on error, no extra command line params
	DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
	DXUTCreateWindow(L"Tutorial08");
	DXUTCreateDevice(D3D_FEATURE_LEVEL_11_1, true, 1280, 768);
	DXUTSetConstantFrameTime(false);
	DXUTMainLoop(); // Enter into the DXUT render loop

	return DXUTGetExitCode();
}