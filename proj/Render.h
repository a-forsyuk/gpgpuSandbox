#pragma once

#include <d3d11.h>
#include <DirectXMath.h>

namespace Render
{
    HRESULT Init(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dContext, uint32_t agentsCount);
    void Release();

    void SetViewProjection(DirectX::CXMMATRIX view, DirectX::FXMMATRIX projection);
    void Clear(ID3D11DeviceContext* pd3dImmediateContext);
    void RenderTerrain(ID3D11DeviceContext* pd3dImmediateContext);
    void RenderAgents(ID3D11DeviceContext* pd3dImmediateContext);
}