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
    int   rayCount;
};

struct Payload {
    float   closestHitT;
    float3  closestHitPos;
    int     hitCount;
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
    // uint rayIndex = DispatchRaysIndex().x;
    uint rayIndex = DispatchRaysIndex().x;

    if (rayIndex >= gPC.rayCount) {
        return;
    }

    float x = -0.5 + rayIndex * (1.0 / max(1.0, (float)(gPC.rayCount - 1)));

    RayDesc ray;
    ray.Origin = float3(x, -1.0, 0.0);
    ray.Direction = normalize(float3(0.0, 1.0, 0.0));
    ray.TMin = 0.0;
    ray.TMax = 2.0;

    Payload p;
    p.closestHitT = 2.0;
    p.closestHitPos = float3(0.0, 0.0, 0.0);
    p.hitCount = 0;

    TraceRay(
        gTLAS,
        RAY_FLAG_NONE,
        0xFF,
        0, 0, 0,
        ray,
        p
    );

    float4 outv;
    if (p.hitCount > 0) {
        outv = float4(p.closestHitPos, p.hitCount);
    } else {
        outv = float4(-1e30, -1e30, -1e30, 0);
    }
    gOutImage[int2(rayIndex, 0)] = outv;
}

// For triangle geometry, the any-hit shader must include the built-in intersection attributes parameter:
[shader("anyhit")]
void aHitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
    float t = RayTCurrent();
    float3 hitPos = WorldRayOrigin() + t * WorldRayDirection();

    // update Payload
    p.hitCount++;
    if (t < p.closestHitT) {
        p.closestHitT = t;
        p.closestHitPos = hitPos;
    }

    // Continue Traversal to find more hits.
    IgnoreHit();
}

[shader("closesthit")]
void chitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
}

[shader("miss")]
void missMain(inout Payload p) {
    // leave payload unchanged
    // p.hitCount = 0;
    // p.closestHitPos = float3(0.0, 0.0, 0.0);
    // p.closestHitT = 2.0;
}