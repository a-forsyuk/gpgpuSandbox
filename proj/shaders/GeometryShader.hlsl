#include "Header.hlsli"

[maxvertexcount(3)]
void main(point PS_INPUT input[1], inout TriangleStream<PS_INPUT> triStream)
{
    float4 pos = input[0].Pos;

    PS_INPUT psInput = (PS_INPUT)0;
    psInput.Pos = mul(pos, ViewProjection);
    psInput.Col = float4(1.0, 1.0, 1.0, 1.0);
    triStream.Append(psInput);

    psInput.Pos = mul(pos + float4(1.5, 0.0, 10.0, 0.0), ViewProjection);
    psInput.Col = input[0].Col;
    triStream.Append(psInput);

    psInput.Pos = mul(pos + float4(-1.5, 0.0, 10.0, 0.0), ViewProjection);
    psInput.Col = input[0].Col;
    triStream.Append(psInput);

    triStream.RestartStrip();
}