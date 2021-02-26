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

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
PS_INPUT VS( VS_INPUT input )
{
    PS_INPUT output = (PS_INPUT)0;
    output.Pos = mul( input.Pos, World );
	//output.Pos = mul( output.Pos, input.Transform);
    output.Pos = mul( output.Pos, View );
    output.Pos = mul( output.Pos, Projection );
    //output.Tex = input.Tex;
	output.Col = input.Col;
    
    return output;
}


[maxvertexcount(3)]   // produce a maximum of 3 output vertices
void GS( point PS_INPUT input[1], inout LineStream<PS_INPUT> triStream)
{
  /*[unroll] for( uint i = 0; i < 3; i++ )
  {*/

    PS_INPUT psInput;
    psInput.Pos = input[0].Pos;
    psInput.Col = input[0].Col;
    triStream.Append(psInput);

    PS_INPUT psInput2;
    psInput2.Pos = input[0].Pos + float4(0.0, 0.0, 20.0, 0.0);
    psInput2.Col = input[0].Col;
    triStream.Append(psInput2);

  //}
  triStream.RestartStrip();
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS(PS_INPUT input) : SV_Target
{
    return input.Col;
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
//PS_Color_INPUT VS_COLOR( VS_Color_INPUT input )
//{
//    PS_Color_INPUT output = (PS_Color_INPUT)0;
//    output.Pos = mul( input.Pos, World );
//    output.Pos = mul( output.Pos, View );
//    output.Pos = mul( output.Pos, Projection );
//    output.Col = input.Col;
//    
//    return output;
//}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
//float4 PS_COLOR( PS_Color_INPUT input) : SV_Target
//{
//    return input.Col;
//}

//--------------------------------------------------------------------------------------
technique11 RenderPositionColor
{
    pass P0
    {
        SetVertexShader( CompileShader(vs_5_0, VS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, PS() ) );
    }
}
