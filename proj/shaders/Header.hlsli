cbuffer COSTANT_BUFFER : register(b0)
{
    row_major matrix ViewProjection;
};

struct VS_INPUT
{
    float4 Pos : POSITION0;
    float4 Col : COLOR0;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Col : COLOR0;
};