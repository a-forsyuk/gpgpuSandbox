#include "Header.hlsli"

PS_INPUT main(VS_INPUT input)
{
    PS_INPUT output = (PS_INPUT)0;

    output.Pos = input.Pos;
    output.Col = input.Col;

    return output;
}