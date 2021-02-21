// CUDA_first.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "CUDA_first.h"
#include <assert.h>
#include "Map.cuh"
#include <iostream>
//#include "PathResult.cuh"
#include "DXUT.h"
#include "DXUTcamera.h"
#include "VertexPositionTexture.h"

#include <DirectXMath.h>

#include "AgentGroup.cuh"
#include "VertexPositionColor.h"
#include "TransformColorInstBatch.h"

#include "d3dx11effect.h"
#include "SDKmisc.h"

#pragma region cuda functions import
extern "C" void InitializeMapNodes(int dimensionX, int dimensionY);
extern "C" void InitializeMap(size_t dimensionX, size_t dimensionY, unsigned int* obstaclesMap);
extern "C" void GetObstacleValue(unsigned int* result);
#pragma endregion cuda functions import

using namespace DirectX;

//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
ID3DX11Effect* g_pEffect = NULL;
ID3D11InputLayout* g_pVertexLayout = NULL;
ID3D11InputLayout* g_pVertexPositionColorLayout = NULL;
ID3DX11EffectTechnique* g_pTechnique = NULL;
ID3DX11EffectTechnique* g_pColorTechnique = NULL;
ID3D11Buffer* g_pVertexBuffer = NULL;
ID3D11Buffer* g_pAxisVertexBuffer = NULL;
ID3D11Buffer* g_pMapVertexBuffer = NULL;
ID3D11Buffer* g_pIndexBuffer = NULL;
ID3D11Buffer* g_pAgentsInstanceData = NULL;
ID3D11ShaderResourceView* g_pTextureRV = NULL;
ID3DX11EffectMatrixVariable* g_pWorldVariable = NULL;
ID3DX11EffectMatrixVariable* g_pViewVariable = NULL;
ID3DX11EffectMatrixVariable* g_pProjectionVariable = NULL;
ID3DX11EffectShaderResourceVariable* g_pDiffuseVariable = NULL;
ID3DX11EffectVectorVariable* g_pMeshColorVariable = NULL;
XMMATRIX g_World;
CModelViewerCamera g_Camera;
AgentGroup* g_pAgentsGroup = NULL;
const UINT AgentsCount = 6144;

int _tmain(int argc, _TCHAR* argv[])
{
	int deviceCount = 0;
	int deviceIndex = 0;
	cudaError_t status = cudaGetDeviceCount(&deviceCount);
	cudaDeviceProp deviceProp = cudaDeviceProp();
	for(int i = 0;i<deviceCount;i++)
	{
		cudaGetDeviceProperties(&deviceProp, i);
		if(deviceProp.major==2)
			deviceIndex = i;
	}
	cudaError_t error = cudaSetDevice(deviceIndex);

	Map::Initialize(make_int2(0,0), make_int2(1600,1600), 40, 40, 100);
	g_pAgentsGroup = new AgentGroup(AgentsCount);
	
	DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
	DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
	DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
	DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
	DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );
	DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );

	DXUTSetCallbackMsgProc( MsgProc );
	DXUTSetCallbackKeyboard( OnKeyboard );
	DXUTSetCallbackFrameMove( OnFrameMove );
	DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );

	DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
	DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
	DXUTCreateWindow( L"Tutorial08" );
	DXUTCreateDevice( D3D_FEATURE_LEVEL_11_1, true, 1280, 768 );
	DXUTSetConstantFrameTime(true);
	DXUTMainLoop(); // Enter into the DXUT render loop

	return DXUTGetExitCode();            


	InitializeMapNodes(10,10);

	#pragma region temp
#pragma endregion temp
}



//--------------------------------------------------------------------------------------
// Reject any D3D10 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo* AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo* DeviceInfo,
	DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext)
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

//	DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
//#if defined( DEBUG ) || defined( _DEBUG )
//	dwShaderFlags |= D3D10_SHADER_DEBUG;
//#endif
	hr = D3DX11CreateEffectFromFile( L"Shader.fx", 0, pd3dDevice, &g_pEffect);
	if( FAILED( hr ) )
	{
		MessageBox( NULL, "The FX file cannot be located.  Please run this executable from the directory that contains the FX file.", "Error", MB_OK );
		V_RETURN( hr );
	}

	g_pTechnique = g_pEffect->GetTechniqueByName( "Render" );
	g_pColorTechnique = g_pEffect->GetTechniqueByName( "RenderPositionColor" );
	g_pWorldVariable = g_pEffect->GetVariableByName( "World" )->AsMatrix();
	g_pViewVariable = g_pEffect->GetVariableByName( "View" )->AsMatrix();
	g_pProjectionVariable = g_pEffect->GetVariableByName( "Projection" )->AsMatrix();
	g_pDiffuseVariable = g_pEffect->GetVariableByName( "txDiffuse" )->AsShaderResource();
	g_pMeshColorVariable = g_pEffect->GetVariableByName( "vMeshColor" )->AsVector();

	// Create the input layout
	D3DX11_PASS_DESC PassDesc;
	g_pTechnique->GetPassByIndex( 0 )->GetDesc( &PassDesc );
	V_RETURN( pd3dDevice->CreateInputLayout( VertexPositionTexture::VertexDescription, VertexPositionTexture::VertexDescriptionElementsCount, PassDesc.pIAInputSignature,
		PassDesc.IAInputSignatureSize, &g_pVertexLayout ) );

	g_pColorTechnique->GetPassByIndex( 0 )->GetDesc( &PassDesc );
	V_RETURN( pd3dDevice->CreateInputLayout( VertexPositionColor::VertexDescription, VertexPositionColor::VertexDescriptionElementsCount, PassDesc.pIAInputSignature,
		PassDesc.IAInputSignatureSize, &g_pVertexPositionColorLayout ) );

	// Create vertex buffer
	VertexPositionTexture vertices[] =
	{
		{ XMFLOAT3{ -0.5f, 0.5f, 0.0f }, XMFLOAT2{ 0.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, 0.5f, 0.0f }, XMFLOAT2{ 1.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, 0.5f, 2.0f }, XMFLOAT2{ 1.0f, 1.0f } },
		{ XMFLOAT3{ -0.5f, 0.5f, 2.0f }, XMFLOAT2{ 0.0f, 1.0f } },

		{ XMFLOAT3{ -0.5f, -0.5f, 0.0f }, XMFLOAT2{ 0.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, -0.5f, 0.0f }, XMFLOAT2{ 1.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, -0.5f, 2.0f }, XMFLOAT2{ 1.0f, 1.0f } },
		{ XMFLOAT3{ -0.5f, -0.5f, 2.0f }, XMFLOAT2{ 0.0f, 1.0f } },

		{ XMFLOAT3{ -0.5f, -0.5f, 2.0f }, XMFLOAT2{ 0.0f, 0.0f } },
		{ XMFLOAT3{ -0.5f, -0.5f, 0.0f }, XMFLOAT2{ 1.0f, 0.0f } },
		{ XMFLOAT3{ -0.5f, 0.5f, 0.0f }, XMFLOAT2{ 1.0f, 1.0f } },
		{ XMFLOAT3{ -0.5f, 0.5f, 2.0f }, XMFLOAT2{ 0.0f, 1.0f } },

		{ XMFLOAT3{ 0.5f, -0.5f, 2.0f }, XMFLOAT2{ 0.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, -0.5f, 0.0f }, XMFLOAT2{ 1.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, 0.5f, 0.0f }, XMFLOAT2{ 1.0f, 1.0f } },
		{ XMFLOAT3{ 0.5f, 0.5f, 2.0f }, XMFLOAT2{ 0.0f, 1.0f } },

		{ XMFLOAT3{ -0.5f, -0.5f, 0.0f }, XMFLOAT2{ 0.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, -0.5f, 0.0f }, XMFLOAT2{ 1.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, 0.5f, 0.0f }, XMFLOAT2{ 1.0f, 1.0f } },
		{ XMFLOAT3{ -0.5f, 0.5f, 0.0f }, XMFLOAT2{ 0.0f, 1.0f } },

		{ XMFLOAT3{ -0.5f, -0.5f, 2.0f }, XMFLOAT2{ 0.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, -0.5f, 2.0f }, XMFLOAT2{ 1.0f, 0.0f } },
		{ XMFLOAT3{ 0.5f, 0.5f, 2.0f }, XMFLOAT2{ 1.0f, 1.0f } },
		{ XMFLOAT3{ -0.5f, 0.5f, 2.0f }, XMFLOAT2{ 0.0f, 1.0f } },
	};

	D3D11_BUFFER_DESC bd;
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof( VertexPositionTexture ) * 24;
	bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	D3D11_SUBRESOURCE_DATA InitData;
	InitData.pSysMem = vertices;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, &InitData, &g_pVertexBuffer ) );

	D3D11_BUFFER_DESC bufferDesc;
	bufferDesc.ByteWidth = AgentsCount * sizeof( TransformColorInstBatch ),
	bufferDesc.Usage = D3D11_USAGE_DYNAMIC,
	bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER,
	bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE,
	bufferDesc.MiscFlags = 0;

	V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, NULL, &g_pAgentsInstanceData ) );
	UINT stride = sizeof(TransformColorInstBatch);
	UINT offset = 0;
	DXUTGetD3D11DeviceContext()->IASetVertexBuffers(1, 1, &g_pAgentsInstanceData, &stride, &offset);

	// Create index buffer
	// Create vertex buffer
	DWORD indices[] =
	{
		3,1,0,
		2,1,3,

		6,4,5,
		7,4,6,

		11,9,8,
		10,9,11,

		14,12,13,
		15,12,14,

		19,17,16,
		18,17,19,

		22,20,21,
		23,20,22
	};

	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof( DWORD ) * 36;
	bd.BindFlags = D3D10_BIND_INDEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	InitData.pSysMem = indices;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, &InitData, &g_pIndexBuffer ) );

	// Set index buffer
	DXUTGetD3D11DeviceContext()->IASetIndexBuffer( g_pIndexBuffer, DXGI_FORMAT_R32_UINT, 0 );

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

	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof( VertexPositionColor ) * 6;
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bd.CPUAccessFlags = 0;
	bd.MiscFlags = 0;
	

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
	hr = DXUTCreateShaderResourceViewFromFile( pd3dDevice, L"seafloor.dds", &g_pTextureRV );
	// Initialize the world matrices
	g_World = XMMatrixIdentity();

	// Initialize the view matrix
	XMVECTORF32 Eye { Map::WorldWidth() / 2.0f, 0.0f, 100.0f };
	XMVECTORF32 At { Map::WorldWidth() / 2.0f, Map::WorldHeight() / 2.0f, 0.0f };
	g_Camera.SetViewParams( Eye, At );

	// Update Variables that never change
	g_pViewVariable->SetMatrix( ( float* )g_Camera.GetViewMatrix() );
	g_pDiffuseVariable->SetResource( g_pTextureRV );

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D10 resources that depend on the back buffer
// Create and set the depth stencil texture if needed
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D10ResizedSwapChain( ID3D10Device* pd3dDevice, IDXGISwapChain* pSwapChain,
	const DXGI_SURFACE_DESC* pBufferSurfaceDesc, void* pUserContext )
{
	// Setup the camera's projection parameters
	float fAspectRatio = static_cast<float>( pBufferSurfaceDesc->Width ) /
		static_cast<float>( pBufferSurfaceDesc->Height );
	g_Camera.SetProjParams( XM_PI / 4, fAspectRatio, 0.1f, 5000.0f );
	g_Camera.SetWindow( pBufferSurfaceDesc->Width, pBufferSurfaceDesc->Height );
	g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );
	g_Camera.SetEnablePositionMovement(true);
	g_pProjectionVariable->SetMatrix( ( float* )g_Camera.GetProjMatrix() );

	return S_OK;
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D10 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10FrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext)
{
	float ClearColor[4] = { 0.1f, 0.1f, 0.1f, 0.1f }; // red, green, blue, alpha
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );

	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D10_CLEAR_DEPTH, 1.0, 0 );

	g_pViewVariable->SetMatrix((float*)g_Camera.GetViewMatrix());
	g_pProjectionVariable->SetMatrix((float*)g_Camera.GetProjMatrix());
	g_pWorldVariable->SetMatrix( ( float* )&g_World );

	// Set vertex buffer
	UINT stride = sizeof( VertexPositionColor );
	UINT offset = 0;
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pAxisVertexBuffer, &stride, &offset );
	pd3dImmediateContext->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_LINELIST);
	pd3dImmediateContext->IASetInputLayout( g_pVertexPositionColorLayout );
	D3DX11_TECHNIQUE_DESC techDesc;
	g_pColorTechnique->GetDesc(&techDesc);
	for( UINT p = 0; p < techDesc.Passes; ++p )
	{
		g_pColorTechnique->GetPassByIndex(p)->Apply( 0 );
		pd3dImmediateContext->Draw(6,0);
	}

	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pMapVertexBuffer, &stride, &offset );
	for( UINT p = 0; p < techDesc.Passes; ++p )
	{
		g_pColorTechnique->GetPassByIndex(p)->Apply( 0, pd3dImmediateContext);
		pd3dImmediateContext->Draw((Map::HeightNodesCount()+1)*2+(Map::WidthNodesCount()+1)*2,0);
	}
	// Set vertex buffer
	stride = sizeof( VertexPositionTexture );
	offset = 0;
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pVertexBuffer, &stride, &offset );
	pd3dImmediateContext->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	pd3dImmediateContext->IASetInputLayout( g_pVertexLayout );
	g_pTechnique->GetDesc( &techDesc );
	size_t agentSize = sizeof(Agent);
	for( UINT p = 0; p < techDesc.Passes; ++p )
	{
			g_pTechnique->GetPassByIndex( p )->Apply( 0 );
			pd3dImmediateContext->DrawIndexedInstanced( 36, AgentsCount, 0, 0, 0 );
	}
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10ReleasingSwapChain( void* pUserContext )
{
}


//--------------------------------------------------------------------------------------
// Release D3D10 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D10DestroyDevice( void* pUserContext )
{
	SAFE_RELEASE( g_pAxisVertexBuffer );
	SAFE_RELEASE( g_pMapVertexBuffer );
	SAFE_RELEASE( g_pVertexBuffer );
	SAFE_RELEASE( g_pAgentsInstanceData);
	SAFE_RELEASE( g_pIndexBuffer );
	SAFE_RELEASE( g_pVertexLayout );
	SAFE_RELEASE( g_pVertexPositionColorLayout );
	SAFE_RELEASE( g_pTextureRV );
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
	TransformColorInstBatch* instanceData;
	DXUTGetD3D11DeviceContext()->Map(g_pAgentsInstanceData, 0, D3D11_MAP_WRITE_DISCARD, NULL, (void**)&instanceData);
	Agent* agents = g_pAgentsGroup->Agents();
	for (int i=0;i<g_pAgentsGroup->AgentsCount();i++)
	{
		Agent agent = agents[i];
		float2 agentPosition = agent.Position();
		XMMATRIX translateMatrix = XMMatrixTranslation(agentPosition.x, agentPosition.y, 0.0f);
		XMMATRIX rotateMatrix = XMMatrixRotationZ(agent.VelocityAngle());
		translateMatrix = XMMatrixMultiply(translateMatrix, g_World);
		translateMatrix = XMMatrixMultiply(rotateMatrix, translateMatrix);
		XMFLOAT4X4 matrix4x4;
		XMStoreFloat4x4(&matrix4x4, translateMatrix);
		instanceData[i].Transform = matrix4x4;
		instanceData[i].Color = XMFLOAT4(agent.color.x, agent.color.y, agent.color.z, 1.0f);
	}
	DXUTGetD3D11DeviceContext()->Unmap(g_pAgentsInstanceData, 0);
	//g_pAgentsGroup->Update(fElapsedTime);
	g_Camera.FrameMove(fElapsedTime);
	/*if(fTime>30.0f)
		DXUTShutdown();*/
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
