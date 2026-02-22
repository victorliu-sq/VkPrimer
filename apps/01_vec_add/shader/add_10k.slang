// add.slang

struct Params
{
    uint N;
    uint totalThreads;
};

[[vk::push_constant]]
Params params;

[[vk::binding(0,0)]] StructuredBuffer<float> A;
[[vk::binding(1,0)]] StructuredBuffer<float> B;
[[vk::binding(2,0)]] RWStructuredBuffer<float> Out;

[numthreads(1024, 1, 1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    for (; i < params.N; i += params.totalThreads)
    {
        Out[i] = A[i] + B[i];
    }
}
