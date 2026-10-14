/* this file is part of example 04_raytracing.py */
import vusvna;

typealias AA3 = AA<3>;

uniform RaytracingAccelerationStructure scene;
uniform RWStructuredBuffer<uint> image;
uniform uint image_width;
uniform uint image_height;
uniform uint ray_marching_steps;
uniform float ray_marching_threshold;
uniform float3 cam_eye;
uniform float3 cam_u;
uniform float3 cam_v;
uniform float3 cam_w;
uniform uint rendering_mode;
uniform float3 background;
uniform StructuredBuffer<float> centers; /* 3 floats per levelset */
uniform StructuredBuffer<float> radii;

struct Payload {
    float3 color;
};

struct Attributes {
    float3 normal;
    uint steps;
};

T f<T : IReal>(T x0, T y0, T z0, float3 origin, float radius) {
    const T x = (x0 - origin.x) / radius;
    const T y = (y0 - origin.y) / radius;
    const T z = (z0 - origin.z) / radius;
    const T xx = x * x;
    const T yy = y * y;
    const T zz = z * z;
    const float c = 0.85 * 0.85;
    const float o = 1.0;
    return ((xx + yy - c) * (xx + yy - c) + (zz - o) * (zz - o))
         * ((yy + zz - c) * (yy + zz - c) + (xx - o) * (xx - o))
         * ((zz + xx - c) * (zz + xx - c) + (yy - o) * (yy - o)) - 0.01;
}

float f(float3 p, float3 origin, float radius) {
    return f(Real(p.x), Real(p.y), Real(p.z), origin, radius).v;
}

/* fake normal, just used for shading */
float3 df(float3 p, float3 origin, float radius) {
    return normalize(p - origin);
}

[shader("intersection")]
void levelset_intersection() {
    const uint primitive_index = PrimitiveIndex();

    const float3 ray_orig = WorldRayOrigin();
    const float3 ray_dir = WorldRayDirection();
    const float ray_tmin = RayTMin();
    const float ray_tmax = RayTCurrent();

    const float3 origin = float3(centers[3 * primitive_index], centers[3 * primitive_index + 1], centers[3 * primitive_index + 2]);
    const float radius = radii[primitive_index];

    /* See Spelunking the Deep: Guaranteed Queries on General Neural Implicit Surfaces via Range Analysis, procedure 2 */
    const float np = 1.5;
    const float nm = 0.5;

    const float f0 = f(ray_orig, origin, radius);
    const float delta = ray_marching_threshold;

    float t = 0.0;
    float sigma = ray_tmax / 10.0;
    for (uint i = 0; i < ray_marching_steps && t <= ray_tmax; ++i) {
        const float3 pc = ray_orig + (t + delta) * ray_dir;
        const float fc = f(pc, origin, radius);
        if (((f0 < 0.0) != (fc < 0.0)) && t >= ray_tmin) {
            Attributes attributes;
            attributes.normal = df(pc, origin, radius);
            attributes.steps = i;
            ReportHit(t, 0, attributes);
            break;
        }
        const float3 c = ray_orig + (t + sigma / 2.0) * ray_dir;
        const float3 v = (sigma / 2.0) * ray_dir;

        const AA3 xv = AA3::variable(0, Interval(c.x - abs(v.x), c.x + abs(v.x)));
        const AA3 yv = AA3::variable(1, Interval(c.y - abs(v.y), c.y + abs(v.y)));
        const AA3 zv = AA3::variable(2, Interval(c.z - abs(v.z), c.z + abs(v.z)));

        if (f(xv, yv, zv, origin, radius).sign() == Sign::UNKNOWN) {
            t += delta;
            sigma *= nm;
        } else {
            t += max(sigma, delta);
            sigma *= np;
        }
    }
}

[shader("closesthit")]
void closest_hit(inout Payload payload, Attributes attributes) {
    if (rendering_mode == 0) {
        payload.color = normalize(attributes.normal) * 0.5 + 0.5;
    } else {
        payload.color = normalize(float3(float(attributes.steps), 0.0, float(ray_marching_steps - 1 - attributes.steps)));
    }
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.color = background;
}

float to_srgb(float c) {
    return c < 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

uint make_color(float3 c) {
    const float3 s = saturate(c);
    const uint r = uint(255.0 * saturate(to_srgb(s.x)) + 0.5);
    const uint g = uint(255.0 * saturate(to_srgb(s.y)) + 0.5);
    const uint b = uint(255.0 * saturate(to_srgb(s.z)) + 0.5);
    return r | (g << 8) | (b << 16) | (255u << 24);
}

[shader("raygeneration")]
void ray_generation() {
    const uint2 idx = DispatchRaysIndex().xy;
    const uint2 dim = DispatchRaysDimensions().xy;

    const float2 d = 2.0 * float2(float(idx.x) / float(dim.x), float(idx.y) / float(dim.y)) - 1.0;

    RayDesc ray;
    ray.Origin = cam_eye;
    ray.Direction = normalize(d.x * cam_u + d.y * cam_v + cam_w);
    ray.TMin = 0.0;
    ray.TMax = 100.0;

    Payload payload = { float3(0.5, 0.5, 0.5) };
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);

    image[idx.y * image_width + idx.x] = make_color(payload.color);
}
