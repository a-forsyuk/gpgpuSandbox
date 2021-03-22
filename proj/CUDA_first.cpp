// CUDA_first.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CUDA_first.h"
#include <assert.h>
#include <iostream>
#include "DXUT.h"
#include "DXUTcamera.h"

#include "Render.h"
#include "Models.h"

#include "SDKmisc.h"

#include "AgentsKernel.cuh"

#include <DirectXMath.h>

using namespace DirectX;

CModelViewerCamera g_Camera;
constexpr uint32_t AgentsCount() { return 10000u; };

constexpr size_t SizeOfPositions() { return AgentsCount() * sizeof(XMFLOAT2); }
XMFLOAT2 positions[AgentsCount()];

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

constexpr size_t SizeOfColors() { return AgentsCount() * sizeof(XMFLOAT4); }
//--------------------------------------------------------------------------------------
// Create any D3D10 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBufferSurfaceDesc,
	void* pUserContext )
{
	HRESULT hr = S_OK;

	Render::Init(pd3dDevice, DXUTGetD3D11DeviceContext(), AgentsCount());

	CUDASystems::GetMapNodesDimensions(&widthNodesCount, &heightNodesCount);

	float worldWidth = 0;
	float worldHeight = 0;
	CUDASystems::GetMapDimensions(&worldWidth, &worldHeight);

	XMVECTORF32 Eye{ worldWidth / 2.0f, 0.0f, 100.0f };
	XMVECTORF32 At{ worldWidth / 2.0f, worldHeight / 2.0f, 0.0f };
	g_Camera.SetViewParams(Eye, At);

	memset(positions, 0, SizeOfPositions());
	XMFLOAT4 colors[AgentsCount()];
	memset(colors, 0, SizeOfColors());

	CUDASystems::MapColors((float*)colors);

	V_RETURN(Models::InitTerrain(pd3dDevice, heightNodesCount, widthNodesCount, worldWidth, worldHeight));
	V_RETURN(Models::InitAgents(pd3dDevice, AgentsCount(), colors, SizeOfColors(), SizeOfPositions()));

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
	CUDASystems::MapPositions((float*)positions);

	Models::UpdateAgents(pd3dImmediateContext, positions, SizeOfPositions());

	Render::Clear(pd3dImmediateContext);
	Render::RenderTerrain(pd3dImmediateContext);
	Render::RenderAgents(pd3dImmediateContext);
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
	Render::SetViewProjection(g_Camera.GetViewMatrix(), g_Camera.GetProjMatrix());
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
	CUDASystems::Init(AgentsCount());

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