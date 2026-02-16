// Slang → SPIR-V → Vulkan KHR Ray Tracing
// Matches C++ bindings:
//   set 0 binding 0 : TLAS
//   set 0 binding 1 : rgba32f storage image
struct PushConstants
{
    int   width;
    int   height;
    float zOrigin;
    float zDir;
};

struct Payload
{
    float3 hitPos;
    int    hit;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> gPC;

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gTLAS;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutImage;

[shader("raygeneration")]
void raygenMain()
{
    uint2 pix = DispatchRaysIndex().xy;

    if (pix.x >= gPC.width || pix.y >= gPC.height)
        return;

    float2 uv = (float2(pix) + 0.5) / float2(gPC.width, gPC.height);
    float2 xy = uv * 2.0 - 1.0;

    RayDesc ray;
    ray.Origin = float3(xy, gPC.zOrigin);
    ray.Direction = normalize(float3(0.0, 0.0, gPC.zDir));
    ray.TMin = 0.0;
    ray.TMax = 1e30;

    Payload p;
    p.hit = 0;
    p.hitPos = float3(0);

    TraceRay(
        gTLAS,
        RAY_FLAG_FORCE_OPAQUE,
        0xFF,
        0, 0, 0,
        ray,
        p
    );

    float4 outv = (p.hit != 0)
        ? float4(p.hitPos, 1.0)
        : float4(-1e30, -1e30, -1e30, 0.0);

    gOutImage[int2(pix)] = outv;
}

[shader("miss")]
void missMain(inout Payload p)
{
    p.hit = 0;
    p.hitPos = float3(0);
}

[shader("closesthit")]
void chitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr)
{
    float t = RayTCurrent();
    float3 hitPos = WorldRayOrigin() + t * WorldRayDirection();

    p.hit = 1;
    p.hitPos = hitPos;
}
