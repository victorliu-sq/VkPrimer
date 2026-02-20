// add.slang
// Minimal compute shader for Vulkan
[[vk::binding(0,0)]] StructuredBuffer<float> A;
[[vk::binding(1,0)]] StructuredBuffer<float> B;
[[vk::binding(2,0)]] RWStructuredBuffer<float> Out;

[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    Out[i] = A[i] + B[i];
}