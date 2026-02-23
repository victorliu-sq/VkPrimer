// lsi_rt.slang
// Slang → SPIR-V → Vulkan KHR Ray Tracing
// Procedural geometry (AABBs) + intersection shader
// Query segments = rays; Base segments = AABB primitives

// -------------------------
// Bindings / layouts
// -------------------------
struct PushConstants
{
    uint  queryEdgeCount;   // number of query edges (= ray count)
    uint  maxOutHits;       // capacity of outHits[]
    uint  _pad0;
    uint  _pad1;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> gPC;

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gTLAS;

// Query map
struct Point2 { float x, y; };
struct Edge   { uint p1_idx, p2_idx; uint _pad0, _pad1; };

[[vk::binding(1, 0)]]
StructuredBuffer<Point2> gQueryPoints;

[[vk::binding(2, 0)]]
StructuredBuffer<Edge> gQueryEdges;

// Base map
[[vk::binding(3, 0)]]
StructuredBuffer<Point2> gBasePoints;

[[vk::binding(4, 0)]]
StructuredBuffer<Edge> gBaseEdges;

// Output (append list)
struct HitRecord
{
    uint  queryEid;
    uint  baseEid;
    float hitx;
    float hity;
};

[[vk::binding(5, 0)]]
RWStructuredBuffer<HitRecord> gOutHits;

[[vk::binding(6, 0)]]
RWStructuredBuffer<uint> gOutCounter;

// -------------------------
// Ray payload + hit attrib
// -------------------------
struct Payload
{
    uint queryEid;
};

// Custom intersection attributes (carried from intersection->anyhit)
struct HitAttrib
{
    uint  baseEid;
    float2 hitXY;
};

// -------------------------
// 2D segment/segment intersection
// Returns (hit, tOnRay, hitPoint)
// Ray segment: O + t*D, t in [0,1]
// Base segment: A->B
// -------------------------
static bool segSegIntersect2D(
    float2 O, float2 D,  // ray segment: [O, O+D]
    float2 A, float2 B,  // base segment: [A, B]
    out float tRay,
    out float2 P)
{
    // Solve: O + tD = A + u(B-A)
    float2 E = B - A;
    float det = D.x * (-E.y) - D.y * (-E.x);  // det([D, -E])

    // Parallel (or nearly parallel)
    if (abs(det) < 1e-12)
        return false;

    float2 rhs = A - O;

    // Cramer's rule:
    // [ D  -E ] [t] = rhs
    //            [u]
    float t = ( rhs.x * (-E.y) - rhs.y * (-E.x)) / det;
    float u = ( D.x * rhs.y - D.y * rhs.x ) / det;

    if (t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0)
        return false;

    tRay = t;
    P = O + t * D;
    return true;
}

// -------------------------
// Raygen: one ray per query edge
// -------------------------
[shader("raygeneration")]
void raygenMain()
{
    uint rayIndex = DispatchRaysIndex().x;
    if (rayIndex >= gPC.queryEdgeCount) return;

    Edge qe = gQueryEdges[rayIndex];
    float2 p1 = float2(gQueryPoints[qe.p1_idx].x, gQueryPoints[qe.p1_idx].y);
    float2 p2 = float2(gQueryPoints[qe.p2_idx].x, gQueryPoints[qe.p2_idx].y);

    float2 O2 = p1;
    float2 D2 = (p2 - p1);

    RayDesc ray;
    ray.Origin = float3(O2.x, O2.y, 0.0);
    ray.Direction = float3(D2.x, D2.y, 0.0);
    ray.TMin = 0.0;
    ray.TMax = 1.0;

    Payload p;
    p.queryEid = rayIndex;

    TraceRay(
        gTLAS,
        RAY_FLAG_NONE,
        0xFF,
        0, 0, 0,
        ray,
        p
    );
}

// -------------------------
// Intersection shader (procedural/AABB):
// PrimitiveIndex() is the AABB primitive id.
// Here we map 1 primitive -> 1 base edge (baseEid == primId).
// If you later bucket many edges per AABB, loop them here.
// -------------------------
[shader("intersection")]
void isectMain(inout HitAttrib attr)
{
    uint primId = PrimitiveIndex();
    uint baseEid = primId;

    // Read queryEid from payload
    // In DXR-style HLSL/Slang, payload is not directly visible here,
    // but we can pass it via RayPayload if your compiler supports it.
    // A portable approach: encode queryEid into rayFlags? Not ideal.
    //
    // Practical Vulkan approach: store queryEid in payload and read it in anyhit;
    // intersection only needs geometry + ray.
    //
    // We'll compute hit for current ray and current base edge and report.

    Edge be = gBaseEdges[baseEid];
    float2 A = float2(gBasePoints[be.p1_idx].x, gBasePoints[be.p1_idx].y);
    float2 B = float2(gBasePoints[be.p2_idx].x, gBasePoints[be.p2_idx].y);

    float3 Or = WorldRayOrigin();
    float3 Dr = WorldRayDirection();
    float2 O2 = float2(Or.x, Or.y);
    float2 D2 = float2(Dr.x, Dr.y);

    float t;
    float2 P;
    if (segSegIntersect2D(O2, D2, A, B, t, P))
    {
        attr.baseEid = baseEid;
        attr.hitXY = P;

        // Report hit at param t (in [0,1]) along the ray segment
        // hitKind can be 0 (unused here)
        ReportHit(t, 0, attr);
    }
}

// -------------------------
// Any-hit: append every intersection, then continue traversal
// -------------------------
[shader("anyhit")]
void anyhitMain(inout Payload p, in HitAttrib attr)
{
    uint idx = 0;
    InterlockedAdd(gOutCounter[0], 1u, idx);

    if (idx < gPC.maxOutHits)
    {
        HitRecord r;
        r.queryEid = p.queryEid;
        r.baseEid  = attr.baseEid;
        r.hitx     = attr.hitXY.x;
        r.hity     = attr.hitXY.y;
        gOutHits[idx] = r;
    }

    // Keep going to find more intersections
    IgnoreHit();
}

// -------------------------
// Closest-hit not used (we want *all* hits via any-hit)
// -------------------------
[shader("closesthit")]
void closesthitMain(inout Payload p, in HitAttrib attr)
{
}

// -------------------------
// Miss: do nothing
// -------------------------
[shader("miss")]
void missMain(inout Payload p)
{
}