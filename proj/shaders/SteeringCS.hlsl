ByteAddressBuffer positions : register(t0);
ByteAddressBuffer targets : register(t1);
RWByteAddressBuffer resPositions : register(u0);

cbuffer constants : register(b0)
{
    float dt;
};

float2 LoadFloat2(ByteAddressBuffer buffer, uint offset)
{
    float x = asfloat(buffer.Load(offset));
    float y = asfloat(buffer.Load(offset + 4));
    return float2(x, y);
}

void StoreFloat2(RWByteAddressBuffer buffer, uint offset, float2 val)
{
    buffer.Store(offset, val.x);
    buffer.Store(offset + 4, val.y);
}

[numthreads(32, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    float2 pos = LoadFloat2(positions, DTid.x * 8);
    float2 target = LoadFloat2(targets, DTid.x * 8);
    pos += normalize(target - pos) * dt;
    StoreFloat2(resPositions, DTid.x * 8, pos);
}