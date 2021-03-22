#include "Header.hlsli"

PS_INPUT main(float2 inputPos : POSITION0, float4 inputColor : COLOR0)
{
    PS_INPUT output = (PS_INPUT)0;

    output.Pos = float4(inputPos, 0, 1);
    output.Col = inputColor;

    return output;
}