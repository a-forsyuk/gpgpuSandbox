#include "Header.hlsli"

[maxvertexcount(3)]
void main(point PS_INPUT input[1], inout TriangleStream<PS_INPUT> triStream)
{
    float4 pos = input[0].Pos;

    matrix vWVP = World;
    vWVP = mul(vWVP, View);
    vWVP = mul(vWVP, Projection);

    PS_INPUT psInput = (PS_INPUT)0;
    psInput.Pos = mul(pos, vWVP);
    psInput.Col = float4(1.0, 1.0, 1.0, 1.0);
    triStream.Append(psInput);

    psInput.Pos = mul(pos + float4(-1.5, -1.5, 10.0, 0.0), vWVP);
    psInput.Col = input[0].Col;
    triStream.Append(psInput);

    psInput.Pos = mul(pos + float4(1.5, 1.5, 10.0, 0.0), vWVP);
    psInput.Col = input[0].Col;
    triStream.Append(psInput);

    triStream.RestartStrip();
}