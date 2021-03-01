technique11 Render
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_5_0, VS_PLAIN() ) );
        SetGeometryShader( CompileShader(gs_5_0, GS() ) );
        SetPixelShader( CompileShader(ps_5_0, PS() ) );
    }
}

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
