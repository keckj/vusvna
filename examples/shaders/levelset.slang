/* this file is part of example 05_levelset.py */
import vusvna;

typealias AA1 = AA<1>;

/* must match the Shape enum of 05_levelset.py */
static const int FLOOR       = 0;
static const int GYROID      = 1;
static const int METABALLS   = 2;
static const int TWISTED_BOX = 3;
static const int KUMMER      = 4;
static const int BARTH       = 5;
static const int HEART       = 6;
static const int TETRAHEDRAL = 7;

/* must match the RenderingMode enum of 05_levelset.py */
static const uint SHADED  = 0;
static const uint NORMALS = 1;
static const uint STEPS   = 2;

/* must match the MarchingMethod enum of 05_levelset.py */
static const uint SIGN_TEST   = 0;
static const uint ROOT_WINDOW = 1;

static const uint RAY_TYPE_RADIANCE = 0;
static const uint RAY_TYPE_SHADOW   = 1;

/* must match the object dtype of 05_levelset.py */
struct Object {
    float3 center;
    float  scale;   /* the shape lives in [-scale, scale]^3 around center, in any rotation around y */
    float  spin;    /* angular velocity around the y axis */
    int    shape;
    float  phase;   /* animation time offset */
    float  pad;
};

uniform RaytracingAccelerationStructure scene;
uniform StructuredBuffer<Object> objects;
uniform RWTexture2D<float4> image;           /* sRGB encoded colors */
uniform RWStructuredBuffer<float4> accum_buffer;
uniform uint image_width;
uniform uint image_height;
uniform uint ray_marching_steps;
uniform float ray_marching_threshold;
uniform float3 cam_eye;
uniform float3 cam_u;
uniform float3 cam_v;
uniform float3 cam_w;
uniform uint rendering_mode;
uniform int subframe_index;
uniform float time;
uniform uint marching_method;
uniform uint samples_per_pixel;


/*----------------------------------------------------------------------------*/
/* Level sets                                                                 */
/*----------------------------------------------------------------------------*/

/* All the level sets are written once as generic functions, evaluated on affine forms to march the rays and
   on Real to refine the roots and compute the normals. Shapes are given in local coordinates, they are
   negative inside and must fit in the cylinder x^2+z^2 <= 1, |y| <= 1. */
interface IShape {
    T eval<T : IReal>(T x, T y, T z);
}

T length3<T : IReal>(T x, T y, T z) {
    return sqrt(sqr(x) + sqr(y) + sqr(z));
}

/* a ball carved by a flowing gyroid lattice */
struct Gyroid : IShape {
    float t;
    T eval<T : IReal>(T x, T y, T z) {
        const float k = 6.0;
        const T X = k * x;
        const T Y = k * y - 0.8 * t;
        const T Z = k * z;
        const T g = sin(X) * cos(Y) + sin(Y) * cos(Z) + sin(Z) * cos(X);
        const float thickness = 0.35 + 0.25 * sin(0.7 * t);
        const T lattice = (abs(g) - thickness) * (1.0 / k);
        const T ball = length3(x, y, z) - 0.95;
        return max(lattice, ball);
    }
}

/* five soft objects with compact polynomial falloff, w(d) = (1 - d^2/R^2)^2 for d < R */
struct Metaballs : IShape {
    float t;
    T blob<T : IReal>(T x, T y, T z, float a, float b, float c, float phase, float R) {
        const float3 center = 0.3 * float3(sin(a * t + phase), cos(b * t + 2.0 * phase), sin(c * t + 3.0 * phase));
        const T d2 = sqr(x - center.x) + sqr(y - center.y) + sqr(z - center.z);
        return sqr(positive_part(1.0 - d2 * (1.0 / (R * R))));
    }
    T eval<T : IReal>(T x, T y, T z) {
        return 0.25 - (blob(x, y, z, 1.1, 0.7, 0.9, 0.0, 0.44) + blob(x, y, z, 0.6, 1.3, 0.5, 1.3, 0.40)
                     + blob(x, y, z, 0.9, 0.4, 1.4, 2.6, 0.36) + blob(x, y, z, 1.7, 0.9, 0.3, 3.9, 0.32)
                     + blob(x, y, z, 0.3, 1.6, 1.1, 5.2, 0.34));
    }
}

/* a rounded box twisted around the y axis */
struct TwistedBox : IShape {
    float t;
    T eval<T : IReal>(T x, T y, T z) {
        const float twist = 2.2 * sin(0.6 * t);
        const T a = twist * y;
        const T c = cos(a);
        const T s = sin(a);
        const T u = c * x - s * z;
        const T w = s * x + c * z;
        const float r = 0.12;
        const T qx = abs(u) - 0.3;
        const T qy = abs(y) - 0.78;
        const T qz = abs(w) - 0.3;
        const T outside = length3(positive_part(qx), positive_part(qy), positive_part(qz));
        const T inside = min(max(qx, max(qy, qz)), 0.0);
        return outside + inside - r;
    }
}

/* Kummer quartic surface clipped by a ball, mu morphs it around its 16 nodes configuration (mu = 1.3) */
struct Kummer : IShape {
    float t;
    T eval<T : IReal>(T x0, T y0, T z0) {
        const float k = 2.4;
        const T x = k * x0;
        const T y = k * z0;
        const T z = k * y0;
        const float mu = 1.3 + 0.2 * sin(0.5 * t);
        const float mu2 = mu * mu;
        const float lambda = (3.0 * mu2 - 1.0) / (3.0 - mu2);
        const float s2 = 1.41421356;
        const T p = 1.0 - z - s2 * x;
        const T q = 1.0 - z + s2 * x;
        const T r = 1.0 + z + s2 * y;
        const T s = 1.0 + z - s2 * y;
        const T f = sqr(sqr(x) + sqr(y) + sqr(z) - mu2) - lambda * ((p * q) * (r * s));
        const T ball = sqr(x) + sqr(y) + sqr(z) - (0.95 * k) * (0.95 * k);
        return max(f, ball);
    }
}

/* Barth sextic (65 nodes for w = 1) clipped by a ball, w breathes around 1 */
struct Barth : IShape {
    float t;
    T eval<T : IReal>(T x0, T y0, T z0) {
        const float k = 1.9;
        const T x = k * x0;
        const T y = k * y0;
        const T z = k * z0;
        const float phi = 1.61803399;
        const float phi2 = phi * phi;
        const float w = 1.0 + 0.12 * sin(0.8 * t);
        const T xx = sqr(x);
        const T yy = sqr(y);
        const T zz = sqr(z);
        const T f = 4.0 * ((phi2 * xx - yy) * (phi2 * yy - zz)) * (phi2 * zz - xx)
                  - (1.0 + 2.0 * phi) * sqr(xx + yy + zz - w * w);
        const T ball = xx + yy + zz - (0.95 * k) * (0.95 * k);
        return max(-f, ball);
    }
}

/* Taubin heart surface (x^2 + 9/4 y^2 + z^2 - 1)^3 - z^3 (x^2 + 9/80 y^2) = 0, beating. The cube root
   of the equation has the same sign and a gradient that does not vanish on the surface where z = 0. */
struct Heart : IShape {
    float t;
    T eval<T : IReal>(T x0, T y0, T z0) {
        const float beat = 1.0 + 0.12 * pow(abs(sin(2.5 * t)), 6.0);
        const float k = 1.35 / beat;
        const T x = k * x0;
        const T y = k * z0;
        const T z = k * y0 + 0.1;
        return sqr(x) + 2.25 * sqr(y) + sqr(z) - 1.0 - z * cbrt(sqr(x) + 0.1125 * sqr(y));
    }
}

/* the tetrahedral quartic r^4 + 8xyz - 10r^2 + c = 0, its handles open and close with c */
struct Tetrahedral : IShape {
    float t;
    T eval<T : IReal>(T x0, T y0, T z0) {
        const float k = 4.0;
        const T x = k * x0;
        const T y = k * y0;
        const T z = k * z0;
        const float c = 20.0 + 4.0 * sin(0.6 * t);
        const T r2 = sqr(x) + sqr(y) + sqr(z);
        return sqr(r2) + 8.0 * ((x * y) * z) - 10.0 * r2 + c;
    }
}

float eval<F : IShape>(F f, float3 p) {
    return f.eval(Real(p.x), Real(p.y), Real(p.z)).v;
}


/* Rotation of an object around its vertical axis, maps world coordinates to local coordinates */
struct Frame {
    float3 center;
    float  inv_scale;
    float  c, s;

    __init(Object object, float t) {
        center = object.center;
        inv_scale = 1.0 / object.scale;
        sincos(object.spin * t, s, c);
    }

    float3 to_local_vector(float3 v) {
        return inv_scale * float3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
    }

    float3 to_local_point(float3 p) {
        return to_local_vector(p - center);
    }

    float3 to_world_normal(float3 n) {
        return normalize(float3(c * n.x + s * n.z, n.y, -s * n.x + c * n.z));
    }
};


/* Finds the first root of f(o + t*d) for t in [t0, t1] up to the threshold delta. Each step evaluates f on a
   segment [t, t+s] of the ray with affine arithmetic, using a single noise symbol e for the ray parameter
   t = t + s/2 + e*s/2, and the step size adapts as in "Spelunking the Deep: Guaranteed Queries on General
   Neural Implicit Surfaces via Range Analysis" (Sharp & Jacobson 2022, procedure 2).

   SIGN_TEST only uses the range of f on the segment: the segment is skipped if f has a certain sign.

   ROOT_WINDOW also uses the affine form f = b + a*e + err of f along the ray: a root is only possible where
   |b + a*e| <= err, which is a window [lo, hi] of e. Everything before the window is skipped and, since
   err shrinks quadratically with the length of the segment around a simple root, the window converges
   quickly to the root. */
bool march<F : IShape>(F f, float3 o, float3 d, float t0, float t1, float delta, out float t_hit, out uint steps) {
    t_hit = t1;
    const bool inside = (eval(f, o + t0 * d) < 0.0);

    float t = t0;
    float sigma = 0.125 * (t1 - t0);
    for (steps = 0; steps < ray_marching_steps && t < t1; ++steps) {
        const float s = min(sigma, t1 - t);
        const AA1 T = AA1::variable(0, Interval(t, t + s));
        const AA1 w = f.eval(o.x + d.x * T, o.y + d.y * T, o.z + d.z * T);

        float lo = -1.0, hi = 1.0;
        if (marching_method == ROOT_WINDOW && !w.is_entire()) {
            const float b = w.b;
            const float a = w.x[0];
            const float R = (inflate(w.radius() + w.err, 1) - abs(a)) * (1.0 + 1e-5);
            if (a != 0.0) {
                const float e0 = (-R - b) / a;
                const float e1 = (+R - b) / a;
                lo = max(lo, min(e0, e1));
                hi = min(hi, max(e0, e1));
            }
        }

        if (w.sign() != Sign::UNKNOWN || !(lo <= hi)) {
            /* no root in [t, t+s] */
            t += s;
            sigma *= 1.5;
            continue;
        }

        const float ta = t + 0.5 * s * (lo + 1.0);
        const float tb = t + 0.5 * s * (hi + 1.0);
        if (tb - ta > delta) {
            /* no root in [t, ta], look closer */
            t = ta;
            sigma = 0.5 * s;
        } else {
            /* root isolated in [ta, tb], refine it if the sign changes */
            if ((eval(f, o + tb * d) < 0.0) != inside) {
                float a = ta, b = tb;
                for (int i = 0; i < 12; ++i) {
                    const float m = 0.5 * (a + b);
                    if ((eval(f, o + m * d) < 0.0) != inside) { b = m; } else { a = m; }
                }
                t_hit = b;
                return true;
            }
            /* tangency or two close roots, move on by at least delta */
            t = max(tb, t + delta);
        }
    }
    return false;
}

/* central differences gradient of f at p */
float3 gradient<F : IShape>(F f, float3 p) {
    const float h = 2e-3;
    return float3(eval(f, p + float3(h, 0, 0)) - eval(f, p - float3(h, 0, 0)),
                  eval(f, p + float3(0, h, 0)) - eval(f, p - float3(0, h, 0)),
                  eval(f, p + float3(0, 0, h)) - eval(f, p - float3(0, 0, h)));
}

bool march_shape(int shape, float t, float3 o, float3 d, float t0, float t1, float delta, out float t_hit, out uint steps) {
    switch (shape) {
        case GYROID:      return march(Gyroid(t), o, d, t0, t1, delta, t_hit, steps);
        case METABALLS:   return march(Metaballs(t), o, d, t0, t1, delta, t_hit, steps);
        case TWISTED_BOX: return march(TwistedBox(t), o, d, t0, t1, delta, t_hit, steps);
        case KUMMER:      return march(Kummer(t), o, d, t0, t1, delta, t_hit, steps);
        case BARTH:       return march(Barth(t), o, d, t0, t1, delta, t_hit, steps);
        case HEART:       return march(Heart(t), o, d, t0, t1, delta, t_hit, steps);
        case TETRAHEDRAL: return march(Tetrahedral(t), o, d, t0, t1, delta, t_hit, steps);
        default:          t_hit = t1; steps = 0; return false;
    }
}

float3 shape_gradient(int shape, float t, float3 p) {
    switch (shape) {
        case GYROID:      return gradient(Gyroid(t), p);
        case METABALLS:   return gradient(Metaballs(t), p);
        case TWISTED_BOX: return gradient(TwistedBox(t), p);
        case KUMMER:      return gradient(Kummer(t), p);
        case BARTH:       return gradient(Barth(t), p);
        case HEART:       return gradient(Heart(t), p);
        case TETRAHEDRAL: return gradient(Tetrahedral(t), p);
        default:          return float3(0.0, 1.0, 0.0);
    }
}

/* entry and exit parameters of the ray in a box, returns false if the ray misses the box */
bool clip_to_box(float3 o, float3 d, float3 lo, float3 hi, inout float t0, inout float t1) {
    const float3 inv_d = 1.0 / d;
    const float3 ta = (lo - o) * inv_d;
    const float3 tb = (hi - o) * inv_d;
    const float3 tmin = min(ta, tb);
    const float3 tmax = max(ta, tb);
    t0 = max(t0, max(tmin.x, max(tmin.y, tmin.z)));
    t1 = min(t1, min(tmax.x, min(tmax.y, tmax.z)));
    return t0 < t1;
}

struct Attributes {
    uint steps;
};

[shader("intersection")]
void levelset_intersection() {
    const Object object = objects[PrimitiveIndex()];

    const float3 ray_orig = WorldRayOrigin();
    const float3 ray_dir = WorldRayDirection();

    Attributes attributes;
    attributes.steps = 0;

    if (object.shape == FLOOR) {
        const float t = (object.center.y - ray_orig.y) / ray_dir.y;
        if (t > RayTMin() && t < RayTCurrent()) { ReportHit(t, 0, attributes); }
        return;
    }

    float t0 = RayTMin();
    float t1 = RayTCurrent();
    const float3 extent = float3(object.scale);
    if (!clip_to_box(ray_orig, ray_dir, object.center - extent, object.center + extent, t0, t1)) { return; }

    const float t = time + object.phase;
    const Frame frame = Frame(object, t);
    const float3 o = frame.to_local_point(ray_orig);
    const float3 d = frame.to_local_vector(ray_dir);

    /* shadow rays only need to know if they are blocked, use a coarser threshold */
    const bool shadow_ray = (RayFlags() & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) != 0;
    const float delta = ray_marching_threshold * (shadow_ray ? 4.0 : 1.0);

    float t_hit;
    uint steps;
    if (march_shape(object.shape, t, o, d, t0, t1, delta, t_hit, steps)) {
        attributes.steps = steps;
        ReportHit(t_hit, 0, attributes);
    }
}


/*----------------------------------------------------------------------------*/
/* Shading                                                                    */
/*----------------------------------------------------------------------------*/

static const float3 sun_direction = float3(-0.45, 0.75, -0.48);

struct RadiancePayload {
    float3 color;
};

struct ShadowPayload {
    uint occluded;
};

float3 sky(float3 dir) {
    const float3 sun = normalize(sun_direction);
    const float h = max(dir.y, 0.0);
    const float3 horizon = float3(0.85, 0.72, 0.62);
    const float3 zenith = float3(0.12, 0.25, 0.55);
    float3 color = lerp(horizon, zenith, pow(h, 0.5));
    const float sd = max(dot(dir, sun), 0.0);
    color += float3(1.0, 0.75, 0.45) * (0.25 * pow(sd, 8.0) + 2.0 * pow(sd, 512.0));
    if (dir.y < 0.0) { color = lerp(color, float3(0.35, 0.3, 0.28), min(-4.0 * dir.y, 1.0)); }
    return color;
}

/* 0 to 1 colormap (blue - cyan - green - yellow - red) */
float3 heatmap(float x) {
    const float t = clamp(x, 0.0, 1.0);
    return saturate(float3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)));
}

float3 albedo(int shape) {
    switch (shape) {
        case GYROID:      return float3(1.00, 0.55, 0.10);
        case METABALLS:   return float3(0.90, 0.15, 0.45);
        case TWISTED_BOX: return float3(0.05, 0.65, 0.60);
        case KUMMER:      return float3(0.20, 0.35, 0.95);
        case BARTH:       return float3(0.45, 0.80, 0.15);
        case HEART:       return float3(0.85, 0.05, 0.08);
        case TETRAHEDRAL: return float3(0.55, 0.25, 0.85);
        default:          return float3(0.8);
    }
}

float3 trace_radiance(float3 origin, float3 direction) {
    RayDesc ray = { origin, 0.0, direction, 1e16 };
    RadiancePayload payload = { float3(0.0) };
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 1, RAY_TYPE_RADIANCE, ray, payload);
    return payload.color;
}

bool trace_occlusion(float3 origin, float3 direction) {
    RayDesc ray = { origin, 0.0, direction, 1e16 };
    ShadowPayload payload = { 1u };
    TraceRay(scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xff,
             0, 1, RAY_TYPE_SHADOW, ray, payload);
    return payload.occluded != 0u;
}

[shader("miss")]
void miss_radiance(inout RadiancePayload payload) {
    payload.color = (rendering_mode == SHADED ? sky(WorldRayDirection()) : float3(0.05));
}

[shader("miss")]
void miss_shadow(inout ShadowPayload payload) {
    payload.occluded = 0u;
}

[shader("closesthit")]
void closest_hit(inout RadiancePayload payload, Attributes attributes) {
    const Object object = objects[PrimitiveIndex()];

    const float3 ray_orig = WorldRayOrigin();
    const float3 ray_dir = WorldRayDirection();
    const float t_hit = RayTCurrent();
    const float3 p = ray_orig + t_hit * ray_dir;

    /* normal and material */
    float3 normal = float3(0.0, 1.0, 0.0);
    float3 color;
    float shininess = 64.0;
    if (object.shape == FLOOR) {
        const float checker = ((int(floor(p.x)) + int(floor(p.z))) & 1) != 0 ? 1.0 : 0.0;
        color = lerp(float3(0.55, 0.53, 0.50), float3(0.75, 0.73, 0.70), checker);
        shininess = 16.0;
    } else {
        const float t = time + object.phase;
        const Frame frame = Frame(object, t);
        normal = frame.to_world_normal(shape_gradient(object.shape, t, frame.to_local_point(p)));
        color = albedo(object.shape);
    }

    /* two sided surfaces: clipped algebraic surfaces show their back faces, tint them */
    if (dot(normal, ray_dir) > 0.0) {
        normal = -normal;
        color = lerp(color, float3(1.0, 0.95, 0.85), 0.6);
    }

    if (rendering_mode == NORMALS) {
        payload.color = normal * 0.5 + 0.5;
        return;
    }
    if (rendering_mode == STEPS) {
        payload.color = (object.shape == FLOOR ? float3(0.1)
                         : heatmap(log(1.0 + float(attributes.steps)) / log(1.0 + float(ray_marching_steps))));
        return;
    }

    /* sun light with hard shadows, hemispheric ambient, Blinn-Phong highlights and Fresnel rim */
    const float3 L = normalize(sun_direction);
    const float3 V = -ray_dir;
    const float3 H = normalize(L + V);
    const float ndotl = dot(normal, L);

    float shadow = 0.0;
    if (ndotl > 0.0) { shadow = trace_occlusion(p + 1e-3 * normal, L) ? 0.0 : 1.0; }

    const float3 sun_color = float3(1.0, 0.92, 0.8) * 2.2;
    const float3 ambient = lerp(float3(0.25, 0.2, 0.18), float3(0.3, 0.4, 0.6), 0.5 + 0.5 * normal.y);
    const float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(normal, V), 0.0), 5.0);
    const float specular = shadow * pow(max(dot(normal, H), 0.0), shininess) * (shininess + 8.0) / 25.0;

    float3 radiance = color * (ambient + sun_color * max(ndotl, 0.0) * shadow);
    radiance += sun_color * specular * 0.25;

    /* the objects reflect the sky, the floor is matte */
    if (object.shape != FLOOR) { radiance += fresnel * 0.5 * sky(reflect(ray_dir, normal)); }

    /* distance fog */
    const float fog = 1.0 - exp(-0.0015 * t_hit * t_hit);
    payload.color = lerp(radiance, sky(ray_dir), fog);
}

float3 to_srgb(float3 c) {
    const float3 s = saturate(c);
    return select(s < 0.0031308, 12.92 * s, 1.055 * pow(s, 1.0 / 2.4) - 0.055);
}

[shader("raygeneration")]
void ray_generation() {
    const uint2 idx = DispatchRaysIndex().xy;
    const uint2 dim = DispatchRaysDimensions().xy;

    /* supersampling: the subpixel offsets follow the R2 low discrepancy sequence, the same for every pixel,
       so that accumulating subframes converges without noise while the scene is still */
    const uint spp = max(samples_per_pixel, 1u);
    float3 color = float3(0.0);
    for (uint s = 0; s < spp; s++) {
        const float k = float(uint(subframe_index) * spp + s);
        const float jx = 0.5 + 0.75487766624669276 * k;
        const float jy = 0.5 + 0.56984029099805326 * k;
        const float2 d = 2.0 * float2((float(idx.x) + frac(jx)) / float(dim.x),
                                      (float(idx.y) + frac(jy)) / float(dim.y)) - 1.0;
        color += trace_radiance(cam_eye, normalize(d.x * cam_u + d.y * cam_v + cam_w));
    }

    /* progressive accumulation, restarted by the host whenever the image changes */
    const uint pixel = idx.y * image_width + idx.x;
    float4 accum = float4(color, float(spp));
    if (subframe_index > 0) { accum += accum_buffer[pixel]; }
    accum_buffer[pixel] = accum;
    color = accum.xyz / accum.w;

    /* vignetting */
    const float2 uv = float2(float(idx.x) / float(dim.x) - 0.5, float(idx.y) / float(dim.y) - 0.5);
    color *= 1.0 - 0.6 * dot(uv, uv);

    /* the camera v axis points up, the rows of the image go down */
    image[uint2(idx.x, dim.y - 1 - idx.y)] = float4(to_srgb(color), 1.0);
}
