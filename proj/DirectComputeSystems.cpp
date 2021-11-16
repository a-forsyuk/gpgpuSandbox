#include "DirectComputeSystems.h"

#include "DoubleBuffer.h"

#include "DXUT.h"

#include <d3d11.h>
#include <DirectXMath.h>

namespace DirectComputeSystems
{
    namespace SteeringCS
    {
#include "SteeringCS.h"
    }

    struct alignas(16) Constants
    {
        float dt;
    };
    Constants constantBufferData{ 0.16f };

    ID3D11ComputeShader* steeringCS = nullptr;
    DoubleBuffer<ID3D11ShaderResourceView> positionsSRV;
    DoubleBuffer<ID3D11UnorderedAccessView> positionsUAV;

    ID3D11Buffer* constantBuffer = NULL;

    HRESULT Init(
        ID3D11Device* device,
        ID3D11DeviceContext* deviceContext,
        ID3D11Buffer* positionsFrontBuffer,
        ID3D11Buffer* positionsBackBuffer,
        DirectX::XMFLOAT2* targets,
        uint32_t agentsCount)
    {
        HRESULT hr = S_OK;

        V_RETURN(device->CreateComputeShader(
            SteeringCS::g_main,
            sizeof(SteeringCS::g_main),
            nullptr,
            &steeringCS)
        );

        D3D11_SHADER_RESOURCE_VIEW_DESC positionsSRVDesc;
        D3D11_BUFFER_DESC cbDesc;

        {
            ZeroMemory(&positionsSRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
            positionsSRVDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
            positionsSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            positionsSRVDesc.Buffer.ElementOffset = 0;
            positionsSRVDesc.Buffer.ElementWidth = sizeof(float) * 2u;

            V_RETURN(device->CreateShaderResourceView(
                positionsFrontBuffer,
                &positionsSRVDesc,
                positionsSRV.GetFrontPtr()));

            ID3D11ShaderResourceView* backSRV = nullptr;
            V_RETURN(device->CreateShaderResourceView(
                positionsBackBuffer,
                &positionsSRVDesc,
                positionsSRV.GetBackPtr()));
        }
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC positionsUAVDesc;
            ZeroMemory(&positionsUAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
            positionsUAVDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
            positionsUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            positionsUAVDesc.Buffer.NumElements = agentsCount;

            //params are swapped because we read from front
            //and write to back buffer at frame 0
            ID3D11UnorderedAccessView* frontPositionsUAV = nullptr;
            V_RETURN(device->CreateUnorderedAccessView(
                positionsFrontBuffer,
                &positionsUAVDesc,
                positionsUAV.GetBackPtr()));

            ID3D11UnorderedAccessView* backPositionsUAV = nullptr;
            V_RETURN(device->CreateUnorderedAccessView(
                positionsBackBuffer,
                &positionsUAVDesc,
                positionsUAV.GetFrontPtr()));
        }

        {
            ID3D11Buffer* targetsBuffer = nullptr;

            ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
            cbDesc.ByteWidth = sizeof(float) * 2u * agentsCount;
            cbDesc.Usage = D3D11_USAGE_IMMUTABLE;
            cbDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

            D3D11_SUBRESOURCE_DATA InitData;
            ZeroMemory(&InitData, sizeof(D3D11_SUBRESOURCE_DATA));
            InitData.pSysMem = &targets;

            V_RETURN(device->CreateBuffer(&cbDesc, &InitData, &targetsBuffer));

            ID3D11ShaderResourceView* targetsSRV = nullptr;

            ZeroMemory(&positionsSRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
            positionsSRVDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
            positionsSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            //positionsSRVDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
            positionsSRVDesc.Buffer.ElementOffset = 0;
            positionsSRVDesc.Buffer.ElementWidth = sizeof(float) * 2u;

            V_RETURN(device->CreateShaderResourceView(targetsBuffer, &positionsSRVDesc, &targetsSRV));
            deviceContext->CSSetShaderResources(1, 1, &targetsSRV);
        }

        {
            ZeroMemory(&cbDesc, sizeof(D3D11_BUFFER_DESC));
            cbDesc.Usage = D3D11_USAGE_DEFAULT;
            cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cbDesc.ByteWidth = sizeof(Constants);

            D3D11_SUBRESOURCE_DATA InitData;
            ZeroMemory(&InitData, sizeof(D3D11_SUBRESOURCE_DATA));
            InitData.pSysMem = &constantBufferData;

            // Create the buffer.
            V_RETURN(device->CreateBuffer(&cbDesc, &InitData, &constantBuffer));
            deviceContext->CSSetConstantBuffers(0, 1, &constantBuffer);
        }

        deviceContext->CSSetShader(steeringCS, nullptr, 0u);

        return hr;
    }

    void Bind(ID3D11DeviceContext* deviceContext, uint32_t agentsCount)
    {
        deviceContext->CSSetShaderResources(0, 1, positionsSRV.GetFrontPtr());
        deviceContext->CSSetUnorderedAccessViews(0, 1, positionsUAV.GetBackPtr(), new uint32_t[]{ agentsCount });
    }

    void Unbind(ID3D11DeviceContext* deviceContext, uint32_t agentsCount)
    {
        deviceContext->CSSetShaderResources(0, 1, new ID3D11ShaderResourceView * [] { nullptr });
        deviceContext->CSSetUnorderedAccessViews(0, 1, new ID3D11UnorderedAccessView * [] { nullptr }, new uint32_t[]{ agentsCount });
    }

    void Update(ID3D11DeviceContext* deviceContext, uint32_t agentsCount, float dt)
    {
        Bind(deviceContext, agentsCount);

        constantBufferData.dt = dt;
        deviceContext->UpdateSubresource(constantBuffer, 0, nullptr, &constantBufferData, 0, 0);

        deviceContext->Dispatch(agentsCount / 32u, 1, 1);

        positionsSRV.Swap();
        positionsUAV.Swap();

        Unbind(deviceContext, agentsCount);
    }
}