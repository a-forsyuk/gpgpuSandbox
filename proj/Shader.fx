//--------------------------------------------------------------------------------------
// File: Tutorial08.fx
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Constant Buffer Variables
//--------------------------------------------------------------------------------------
//Texture2D txDiffuse;
//SamplerState samLinear
//{
//    Filter = MIN_MAG_MIP_LINEAR;
//    AddressU = Wrap;
//    AddressV = Wrap;
//};

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
    float4 Pos : POSITION;
    float2 Tex : TEXCOORD;
	float4 Col : COLOR0;
	row_major float4x4 Transform : mTransform;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
	float4 Col : COLOR0;
};

struct VS_Color_INPUT
{
    float4 Pos : POSITION;
    float4 Col : COLOR;
};

struct PS_Color_INPUT
{
    float4 Pos : SV_POSITION;
    float4 Col : COLOR0;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
PS_INPUT VS( VS_INPUT input )
{
    PS_INPUT output = (PS_INPUT)0;
    output.Pos = mul( input.Pos, World );
	output.Pos = mul( output.Pos, input.Transform);
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( output.Pos, Projection );
    output.Tex = input.Tex;
	output.Col = input.Col;
    
    return output;
}


[maxvertexcount(3)]   // produce a maximum of 3 output vertices
void GS( triangle PS_INPUT input[3], inout TriangleStream<PS_INPUT> triStream)
{
  PS_INPUT psInput;
  //float3 faceEdgeA = input[1].Pos - input[0].Pos;
  //float3 faceEdgeB = input[2].Pos - input[0].Pos;
  //float3 faceNormal = normalize( cross(faceEdgeA, faceEdgeB) );
  //float3 centerPos = (input[0].Pos.xyz + input[1].Pos.xyz + input[2].Pos.xyz)/3.0;
  //psInput.Pos = float4(centerPos.x, centerPos.y, centerPos.z, 0);
  //psInput.Col = float4(1.0f,1.0f,1.0f,1.0f);
  //psInput.Tex = float2(0.0f,0.0f);
  //triStream.Append(psInput);
  //psInput.Pos = float4(faceNormal, 0);
  //triStream.Append(psInput);
  //triStream.RestartStrip();
  [unroll] for( uint i = 0; i < 3; i++ )
  {
    psInput.Pos = input[i].Pos;
    psInput.Col = input[i].Col;
	psInput.Tex = input[i].Tex;
    triStream.Append(psInput);
  }
  triStream.RestartStrip();
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS(PS_INPUT input) : SV_Target
{
    return input.Col;
    //txDiffuse.Sample(samLinear, input.Tex)* input.Col;// * vMeshColor;
}


//--------------------------------------------------------------------------------------
technique11 Render
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_5_0, VS() ) );
        SetGeometryShader( CompileShader(gs_5_0, GS() ) );
        SetPixelShader( CompileShader(ps_5_0, PS() ) );
    }
}

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
PS_Color_INPUT VS_COLOR( VS_Color_INPUT input )
{
    PS_Color_INPUT output = (PS_Color_INPUT)0;
    output.Pos = mul( input.Pos, World );
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( output.Pos, Projection );
    output.Col = input.Col;
    
    return output;
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS_COLOR( PS_Color_INPUT input) : SV_Target
{
    return input.Col;
}

//--------------------------------------------------------------------------------------
technique11 RenderPositionColor
{
    pass P0
    {
        SetVertexShader( CompileShader(vs_5_0, VS_COLOR() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, PS_COLOR() ) );
    }
}
