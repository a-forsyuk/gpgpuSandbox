cbuffer cbNeverChanges
{
    matrix View;
};

cbuffer cbChangeOnResize
{
    matrix Projection;
};

cbuffer cbChangesEveryFrame
{
    matrix World;
    float4 vMeshColor;
};

struct VS_INPUT
{
    float4 Pos : POSITION0;
    float4 Col : COLOR0;
    //row_major float4x4 Transform : mTransform;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Col : COLOR0;
};