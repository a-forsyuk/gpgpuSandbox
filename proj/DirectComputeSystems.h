#pragma once

#include <winerror.h>
#include <stdint.h>

struct ID3D11Device;
struct ID3D11Buffer;
struct ID3D11DeviceContext;
namespace DirectX
{
    struct XMFLOAT2;
}

namespace DirectComputeSystems
{
    HRESULT Init(
        ID3D11Device* device, 
        ID3D11DeviceContext* deviceContext, 
        ID3D11Buffer* positionsFronBuffer, 
        ID3D11Buffer* positionBackBuffer,
        DirectX::XMFLOAT2* targets,
        uint32_t agentsCount);

    void Bind(ID3D11DeviceContext* deviceContext, uint32_t agentsCount);

    void Update(ID3D11DeviceContext* deviceContext, uint32_t agentsCount, float dt);

    void UnBind(ID3D11DeviceContext* deviceContext, uint32_t agentsCount);
}