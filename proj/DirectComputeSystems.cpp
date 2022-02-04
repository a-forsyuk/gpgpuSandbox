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

    ID3D11Buffer* targetsBuffer = nullptr;
    ID3D11ShaderResourceView* targetsSRV = nullptr;

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
        ZeroMemory(&positionsSRVDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
        positionsSRVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
        positionsSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        //positionsSRVDesc.Buffer.ElementOffset = 0;
        //positionsSRVDesc.Buffer.ElementWidth = sizeof(float) * 2u;
        positionsSRVDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        positionsSRVDesc.BufferEx.FirstElement = 0;
        positionsSRVDesc.BufferEx.NumElements = agentsCount * 2u;

        D3D11_BUFFER_DESC cbDesc;

        {
            V_RETURN(device->CreateShaderResourceView(
                positionsFrontBuffer,
                &positionsSRVDesc,
                positionsSRV.GetFrontPtr()));

            V_RETURN(device->CreateShaderResourceView(
                positionsBackBuffer,
                &positionsSRVDesc,
                positionsSRV.GetBackPtr()));
        }
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC positionsUAVDesc;
            ZeroMemory(&positionsUAVDesc, sizeof(D3D11_UNORDERED_ACCESS_VIEW_DESC));
            positionsUAVDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            positionsUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            positionsUAVDesc.Buffer.NumElements = agentsCount * 2u;
            positionsUAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

            //params are swapped because we read from front
            //and write to back buffer at frame 0
            V_RETURN(device->CreateUnorderedAccessView(
                positionsFrontBuffer,
                &positionsUAVDesc,
                positionsUAV.GetFrontPtr()));

            V_RETURN(device->CreateUnorderedAccessView(
                positionsBackBuffer,
                &positionsUAVDesc,
                positionsUAV.GetBackPtr()));
        }

        {
            D3D11_BUFFER_DESC cbDesc1;
            ZeroMemory(&cbDesc1, sizeof(D3D11_BUFFER_DESC));
            cbDesc1.ByteWidth = sizeof(float) * 2u * agentsCount;
            cbDesc1.Usage = D3D11_USAGE_IMMUTABLE;
            cbDesc1.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            cbDesc1.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

            D3D11_SUBRESOURCE_DATA InitData;
            ZeroMemory(&InitData, sizeof(D3D11_SUBRESOURCE_DATA));

            InitData.pSysMem = &targets;

            V_RETURN(device->CreateBuffer(&cbDesc1, &InitData, &targetsBuffer));

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
        deviceContext->CSSetShaderResources(1, 1, &targetsSRV);
        deviceContext->CSSetUnorderedAccessViews(0, 1, positionsUAV.GetBackPtr(), new uint32_t[]{ agentsCount * 2u });
    }

    void Unbind(ID3D11DeviceContext* deviceContext, uint32_t agentsCount)
    {
        deviceContext->CSSetShaderResources(0, 2, new ID3D11ShaderResourceView * [] { nullptr, nullptr });
        deviceContext->CSSetUnorderedAccessViews(0, 1, new ID3D11UnorderedAccessView * [] { nullptr }, new uint32_t[]{ agentsCount * 2u });
    }

    void Update(ID3D11DeviceContext* deviceContext, uint32_t agentsCount, float dt)
    {
        Bind(deviceContext, agentsCount);

        constantBufferData.dt = dt;
        deviceContext->UpdateSubresource(constantBuffer, 0, nullptr, &constantBufferData, 0, 0);

        deviceContext->Dispatch(agentsCount / 32, 1, 1);

        Unbind(deviceContext, agentsCount);

        positionsSRV.Swap();
        positionsUAV.Swap();
    }
}