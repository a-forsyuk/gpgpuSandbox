// CUDA_first.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CUDA_first.h"
#include <assert.h>
#include <iostream>
#include "DXUT.h"
#include "DXUTcamera.h"

//#include "VertexPositionTexture.h"
#include "VertexPositionColor.h"
#include "TransformColorInstBatch.h"

#include "d3dx11effect.h"
#include "SDKmisc.h"

#include "Map.cuh"
#include "Agent.cuh"

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
ID3DX11Effect* g_pEffect = nullptr;
ID3D11InputLayout* g_pVertexLayout = nullptr;
ID3D11InputLayout* g_pVertexPositionColorLayout = nullptr;
ID3DX11EffectTechnique* g_pTechnique = nullptr;
ID3DX11EffectTechnique* g_pColorTechnique = nullptr;
ID3D11Buffer* g_pAxisVertexBuffer = nullptr;
ID3D11Buffer* g_pMapVertexBuffer = nullptr;
ID3D11Buffer* g_pAgentsInstanceData = nullptr;
ID3DX11EffectMatrixVariable* g_pWorldVariable = nullptr;
ID3DX11EffectMatrixVariable* g_pViewVariable = nullptr;
ID3DX11EffectMatrixVariable* g_pProjectionVariable = nullptr;
ID3DX11EffectVectorVariable* g_pMeshColorVariable = nullptr;

ID3D11VertexShader* g_plainVertexShader = nullptr;

XMMATRIX g_World;
CModelViewerCamera g_Camera;
AgentGroup* g_pAgentsGroup = nullptr;
const UINT AgentsCount = 6144;
VertexPositionColor* instanceData = nullptr;

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

	V_RETURN(pd3dDevice->CreateVertexShader(PlainVs::g_main, sizeof(PlainVs::g_main), nullptr, &g_plainVertexShader));

	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
	// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
	// Setting this flag improves the shader debugging experience, but still allows 
	// the shaders to be optimized and to run exactly the way they will run in 
	// the release configuration of this program.
	dwShaderFlags |= D3DCOMPILE_DEBUG;

	// Disable optimizations to further improve shader debugging
	dwShaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
	ID3DBlob* pErrorBlob = nullptr;
	hr = D3DX11CompileEffectFromFile(L"Shader.fx", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, dwShaderFlags, 0, pd3dDevice, &g_pEffect, &pErrorBlob);
	if( FAILED( hr ) )
	{
		MessageBox( NULL, L"The FX file cannot be located.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
		V_RETURN( hr );
	}

	g_pTechnique = g_pEffect->GetTechniqueByName( "Render" );
	g_pColorTechnique = g_pEffect->GetTechniqueByName( "RenderPositionColor" );
	g_pWorldVariable = g_pEffect->GetVariableByName( "World" )->AsMatrix();
	g_pViewVariable = g_pEffect->GetVariableByName( "View" )->AsMatrix();
	g_pProjectionVariable = g_pEffect->GetVariableByName( "Projection" )->AsMatrix();
	g_pMeshColorVariable = g_pEffect->GetVariableByName( "vMeshColor" )->AsVector();

	// Create the input layout
	D3DX11_PASS_DESC PassDesc;
	g_pTechnique->GetPassByIndex( 0 )->GetDesc( &PassDesc );
	V_RETURN( pd3dDevice->CreateInputLayout( VertexPositionColor::VertexDescription, VertexPositionColor::VertexDescriptionElementsCount, PassDesc.pIAInputSignature,
		PassDesc.IAInputSignatureSize, &g_pVertexLayout ) );

	g_pColorTechnique->GetPassByIndex( 0 )->GetDesc( &PassDesc );
	V_RETURN( pd3dDevice->CreateInputLayout( VertexPositionColor::VertexDescription, VertexPositionColor::VertexDescriptionElementsCount, PassDesc.pIAInputSignature,
		PassDesc.IAInputSignatureSize, &g_pVertexPositionColorLayout ) );

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

	///Map///
	int verticesCount = (Map::HeightNodesCount() + 1) * 2 + (Map::WidthNodesCount() + 1) * 2;
	VertexPositionColor* mapLinesVertices = new VertexPositionColor[verticesCount];

	for (int i=0;i<verticesCount;i++)
	{
		mapLinesVertices[i].color = XMFLOAT4{ 0.2f,0.2f,0.2f,0.1f };
	}

	int i = 0;

	mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
	mapLinesVertices[i++].position = XMFLOAT3{ Map::WorldWidth(), 0.0f, 0.0f };

	for (int heightCounter = 1; heightCounter<Map::HeightNodesCount() + 1; heightCounter++)
	{
		mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, Map::WorldHeight() / Map::HeightNodesCount() * heightCounter, 0.0f };
		mapLinesVertices[i++].position = XMFLOAT3{ Map::WorldWidth() , Map::WorldHeight() / Map::HeightNodesCount() * heightCounter, 0.0f };
	}

	mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, 0.0f, 0.0f };
	mapLinesVertices[i++].position = XMFLOAT3{ 0.0f, Map::WorldHeight(), 0.0f };

	for (int widthCounter = 1;widthCounter<Map::WidthNodesCount()+1;widthCounter++)
	{
		mapLinesVertices[i++].position = XMFLOAT3{ Map::WorldWidth() / Map::WidthNodesCount() * widthCounter, 0.0f, 0.0f };
		mapLinesVertices[i++].position = XMFLOAT3{ Map::WorldWidth() / Map::WidthNodesCount() * widthCounter, Map::WorldHeight(), 0.0f };
	}

	bd.ByteWidth = sizeof( VertexPositionColor ) * verticesCount;

	InitData.pSysMem = mapLinesVertices;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, &InitData, &g_pMapVertexBuffer ) );

	// Load the Texture
	//hr = DXUTCreateShaderResourceViewFromFile( pd3dDevice, L"seafloor.dds", &g_pTextureRV );
	// Initialize the world matrices
	g_World = XMMatrixIdentity();

	// Initialize the view matrix
	XMVECTORF32 Eye { Map::WorldWidth() / 2.0f, 0.0f, 100.0f };
	XMVECTORF32 At { Map::WorldWidth() / 2.0f, Map::WorldHeight() / 2.0f, 0.0f };
	g_Camera.SetViewParams( Eye, At );

	// Update Variables that never change

	XMFLOAT4X4 viewMatrix;
	XMStoreFloat4x4(&viewMatrix, g_Camera.GetViewMatrix());

	g_pViewVariable->SetMatrix( (float*)viewMatrix.m );
	//g_pDiffuseVariable->SetResource( g_pTextureRV );

	instanceData = new VertexPositionColor[g_pAgentsGroup->AgentsCount()];

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
	XMFLOAT4X4 projMatrix;
	XMStoreFloat4x4(&projMatrix, g_Camera.GetProjMatrix());
	g_pProjectionVariable->SetMatrix( ( float* )projMatrix.m );

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

	XMFLOAT4X4 viewMatrix;
	XMStoreFloat4x4(&viewMatrix, g_Camera.GetViewMatrix());
	g_pViewVariable->SetMatrix((float*)viewMatrix.m);

	XMFLOAT4X4 projMatrix;
	XMStoreFloat4x4(&projMatrix, g_Camera.GetProjMatrix());
	g_pProjectionVariable->SetMatrix((float*)projMatrix.m);
	g_pWorldVariable->SetMatrix( ( float* )&g_World );

	// Set vertex buffer
	UINT stride = sizeof( VertexPositionColor );
	UINT offset = 0;
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pAxisVertexBuffer, &stride, &offset );
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
	pd3dImmediateContext->IASetInputLayout( g_pVertexPositionColorLayout );
	D3DX11_TECHNIQUE_DESC techDesc;
	g_pColorTechnique->GetDesc(&techDesc);
	for( UINT p = 0; p < techDesc.Passes; ++p )
	{
		g_pColorTechnique->GetPassByIndex(p)->Apply( 0 , pd3dImmediateContext);
		pd3dImmediateContext->Draw(6,0);
	}

	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pMapVertexBuffer, &stride, &offset );
	for( UINT p = 0; p < techDesc.Passes; ++p )
	{
		g_pColorTechnique->GetPassByIndex(p)->Apply( 0, pd3dImmediateContext);
		pd3dImmediateContext->Draw((Map::HeightNodesCount()+1)*2+(Map::WidthNodesCount()+1)*2,0);
	}

	D3D11_MAPPED_SUBRESOURCE mappedSubresource;
	ZeroMemory(&mappedSubresource, sizeof(D3D11_MAPPED_SUBRESOURCE));
	//mappedSubresource.pData = &instanceData;
	//mappedSubresource.RowPitch = 0;
	//mappedSubresource.DepthPitch = 0;
	HRESULT res = DXUTGetD3D11DeviceContext()->Map(g_pAgentsInstanceData, 0, D3D11_MAP_WRITE_DISCARD, NULL, &mappedSubresource);
	Agent* agents = g_pAgentsGroup->Agents();
	for (int i = 0; i < g_pAgentsGroup->AgentsCount(); i++)
	{
		Agent agent = agents[i];
		instanceData[i].position.x = agent.Position().x;
		instanceData[i].position.y = agent.Position().y;
		instanceData[i].position.z = 0.0f;
		instanceData[i].color.x = agent.color.x;
		instanceData[i].color.y = agent.color.y;
		instanceData[i].color.z = agent.color.z;
	}
	memcpy(mappedSubresource.pData, instanceData, sizeof(VertexPositionColor) * AgentsCount);
	DXUTGetD3D11DeviceContext()->Unmap(g_pAgentsInstanceData, 0);
	stride = sizeof(VertexPositionColor);
	offset = 0;
	DXUTGetD3D11DeviceContext()->IASetVertexBuffers(0, 1, &g_pAgentsInstanceData, &stride, &offset);
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
	pd3dImmediateContext->IASetInputLayout( g_pVertexLayout );
	g_pTechnique->GetDesc( &techDesc );
	for( UINT p = 0; p < techDesc.Passes; ++p )
	{
			g_pTechnique->GetPassByIndex( p )->Apply( 0, pd3dImmediateContext);
			pd3dImmediateContext->Draw( AgentsCount, 0 );
	}

	// Set vertex buffer
	//stride = sizeof( VertexPositionTexture );
	//offset = 0;
	//pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pVertexBuffer, &stride, &offset );
	//pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	//pd3dImmediateContext->IASetInputLayout( g_pVertexLayout );
	//g_pTechnique->GetDesc( &techDesc );
	//for( UINT p = 0; p < techDesc.Passes; ++p )
	//{
	//		g_pTechnique->GetPassByIndex( p )->Apply( 0, pd3dImmediateContext);
	//		pd3dImmediateContext->DrawIndexedInstanced( 36, AgentsCount, 0, 0, 0 );
	//}
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
	//SAFE_RELEASE( g_pVertexBuffer );
	SAFE_RELEASE( g_pAgentsInstanceData);
	//SAFE_RELEASE( g_pIndexBuffer );
	SAFE_RELEASE( g_pVertexLayout );
	SAFE_RELEASE( g_pVertexPositionColorLayout );
	//SAFE_RELEASE( g_pTextureRV );
	SAFE_RELEASE( g_pEffect );
	cudaDeviceReset();
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
	//instanceData = new TransformColorInstBatch[g_pAgentsGroup->AgentsCount()];
	g_pAgentsGroup->Update(fElapsedTime);
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
	int deviceCount = 0;
	int deviceIndex = 0;
	cudaError_t status = cudaGetDeviceCount(&deviceCount);
	cudaDeviceProp deviceProp = cudaDeviceProp();
	for (int i = 0; i < deviceCount; i++)
	{
		cudaGetDeviceProperties(&deviceProp, i);
		if (deviceProp.major == 2)
			deviceIndex = i;
	}
	cudaError_t error = cudaSetDevice(deviceIndex);

	Map::Initialize(make_float2(0.0f, 0), make_float2(1600.0f, 1600.0f), 40, 40, 100);
	g_pAgentsGroup = new AgentGroup(AgentsCount);

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
	DXUTSetConstantFrameTime(true);
	DXUTMainLoop(); // Enter into the DXUT render loop

	return DXUTGetExitCode();
}