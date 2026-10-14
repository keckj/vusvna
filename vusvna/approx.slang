/* Rigorous affine approximations of elementary functions.
 *
 * For a function f and an input affine form x = x0 + rho, where x0 is the affine part (whose range
 * is X) and rho the independent error term (|rho| <= err), an approximation is a tuple
 * (alpha, zeta, delta, err') such that:
 *     f(x0) - alpha*x0  belongs to  [zeta - delta, zeta + delta]   for all x0 in X  (residual)
 *     |f(x0 + rho) - f(x0)| <= err'                                                 (propagated error)
 *
 * The range of the residual g(x) = f(x) - alpha*x is computed with outward rounding by splitting X
 * into pieces where f is convex or concave: g is then bounded by its values at the endpoints on one
 * side and by a tangent of f on the other side. Pieces where f has an inflection point (known up to
 * rounding) are thin and are bounded with interval arithmetic assuming f is monotonic on them.
 *
 * The functions only rely on rigorous enclosures of f and f' at points (F and DF). They are built from
 * the error bounds of the Vulkan specification (exp, exp2, log, log2, atan), or computed by vusvna
 * when the specification is too loose (sin and cos, guaranteed to 2^-11 only) or the function does not
 * exist in SPIR-V (cbrt, exp10, sinpi, cospi). The approximations f and f' only choose the slopes and tangent points, their
 * accuracy does not matter.
 */
implementing vusvna;

public struct Approx {
    public float alpha;
    public float zeta;
    public float delta;
    public float err;
    public bool entire;

    public static Approx make_entire() {
        Approx r;
        r.alpha = 0.0; r.zeta = 0.0; r.delta = 0.0; r.err = HUGE; r.entire = true;
        return r;
    }

    /* exact affine function alpha*x + zeta, plus err */
    public static Approx make_affine(float alpha, float zeta, float err) {
        Approx r;
        r.alpha = alpha; r.zeta = zeta; r.delta = 0.0; r.err = err; r.entire = !(err <= LIMIT);
        return r;
    }

    /* constant function whose values lie in [lo, hi] */
    public static Approx make_range(float lo, float hi) {
        return from_residual(0.0, Interval(lo, hi), 0.0);
    }
};

/* bound of the propagated error for a function with lipschitz constant lip */
float propagate(float lip, float err) {
    if (err == 0.0) { return 0.0; }
    if (!(lip <= LIMIT)) { return HUGE; }
    return mul_up(lip, err);
}

/* approximation from the slope alpha and an enclosure g of the residual */
Approx from_residual(float alpha, Interval g, float err) {
    if (!is_bounded(alpha) || !g.is_bounded() || !(err <= LIMIT)) {
        return Approx::make_entire();
    }
    Approx r;
    r.alpha = alpha;
    r.zeta = 0.5 * g.a + 0.5 * g.b;
    r.delta = f_max(sub_up(g.b, r.zeta), sub_up(r.zeta, g.a));
    r.err = err;
    r.entire = false;
    return r;
}

/* the affine part range enlarged by the error term */
Interval widen(Interval X, float err) { return Interval(sub_down(X.a, err), add_up(X.b, err)); }

/* x^n for an integer n >= 0 */
Interval ipow(Interval x, int n) {
    Interval result = point(1.0);
    for (int i = 0; i < 64 && n > 0; ++i) {
        if ((n & 1) != 0) { result = result * x; }
        n >>= 1;
        if (n > 0) { x = sqr(x); }
    }
    return result;
}

float fpown(float x, int n) {
    const bool inverse = (n < 0);
    int m = (inverse ? -n : n);
    float result = 1.0;
    for (int i = 0; i < 64 && m > 0; ++i) {
        if ((m & 1) != 0) { result *= x; }
        m >>= 1;
        if (m > 0) { x *= x; }
    }
    return (inverse ? 1.0 / result : result);
}

/* positive modulo */
int pmod(int m, int n) {
    const int r = m % n;
    return (r < 0 ? r + n : r);
}


/*** rigorous enclosures of elementary functions at a point ***/

static const float EXP_MAX = 41.0;      /* exp(41) < 2^60 */
static const float EXP2_MAX = 59.0;
static const float EXP_TINY = 9.629649721936179e-35;  /* 2^-113 > exp(-79) */
static const float EXP2_TINY = 2.407412430484045e-35;  /* 2^-115 */

/* exp and exp2 are within 3 + 2|x| ulps */
public Interval exp_enclosure(float x) {
    if (x > EXP_MAX) { return Interval::entire(); }
    if (x < -79.0) { return Interval(0.0, EXP_TINY); }
    return Interval(f_max(down_ulps(exp(x), 3.0 + 2.0 * f_abs(x)), 0.0), up_ulps(exp(x), 3.0 + 2.0 * f_abs(x)));
}

public Interval exp2_enclosure(float x) {
    if (x > EXP2_MAX) { return Interval::entire(); }
    if (x < -115.0) { return Interval(0.0, EXP2_TINY); }
    return Interval(f_max(down_ulps(exp2(x), 3.0 + 2.0 * f_abs(x)), 0.0), up_ulps(exp2(x), 3.0 + 2.0 * f_abs(x)));
}

/* log and log2 are within 3 ulps outside [0.5, 2], within 2^-21 inside, x > 0 is expected */
static const float LOG_ABS_ERR = 4.76837158203125e-7;  /* 2^-21 */
static const float LOG_SMALL = 7.888609052210118e-31;  /* 2^-100 */

public Interval log_enclosure(float x) {
    if (!(x >= LOG_SMALL)) { return Interval(-HUGE, -69.0); }  /* log(2^-100) < -69.3 */
    const float r = log(x);
    if (x >= 0.5 && x <= 2.0) { return enclose_abs(r, LOG_ABS_ERR); }
    return enclose_ulps(r, 3.0);
}

public Interval log2_enclosure(float x) {
    if (!(x >= LOG_SMALL)) { return Interval(-HUGE, -100.0); }
    const float r = log2(x);
    if (x >= 0.5 && x <= 2.0) { return enclose_abs(r, LOG_ABS_ERR); }
    return enclose_ulps(r, 3.0);
}

/* atan is within 4096 ulps */
static const float ATAN_ULPS = 4096.0;
public Interval atan_enclosure(float x) {
    const Interval r = enclose_ulps(atan(x), ATAN_ULPS);
    return intersect(r, Interval(-0.5 * PI_HI, 0.5 * PI_HI));
}

/* x^y = exp2(y log2(x)) for x >= 0, computed by vusvna from the bounds of exp2 and log2 */
public Interval pow_enclosure(float x, float y) {
    if (y == 0.0) { return point(1.0); }
    if (!(x >= LOG_SMALL)) {
        if (y < 0.0) { return Interval::entire(); }
        /* x^y is increasing, x^y <= 2^(-100 y) */
        return Interval(0.0, (y >= 1.0 ? LOG_SMALL : exp2_enclosure(up(-100.0 * y)).b));
    }
    const Interval L = log2_enclosure(x);
    const float l = 0.5 * L.a + 0.5 * L.b;
    const float el = f_max(l - L.a, L.b - l) * 1.0000002384185791;
    const float t = y * l;
    /* |t - y log2(x)| <= et */
    const float et = up(mul_up(f_abs(y), el) + (f_abs(t) * 1.1920928955078125e-7 + 4.70197740328915e-38));
    if (!(et <= 0.5) || t > EXP2_MAX) { return Interval::entire(); }
    const Interval R = exp2_enclosure(t);
    /* 2^-et >= 1 - et and 2^et <= 1 + et for 0 <= et <= 1/2 */
    return Interval(f_max(mul_down(R.a, sub_down(1.0, et)), 0.0), mul_up(R.b, add_up(1.0, et)));
}

/* x^3 bounds for x >= 0 */
float cube_up(float x) { return mul_up(mul_up(x, x), x); }
float cube_down(float x) { return f_max(mul_down(mul_down(x, x), x), 0.0); }

/* cbrt(x), an approximation refined by a Newton step and validated by cubing its bounds */
public Interval cbrt_enclosure(float x) {
    const float a = f_abs(x);
    if (!(a <= LIMIT)) { return Interval::entire(); }
    Interval R;
    if (a <= 1.0e-30) {
        R = Interval(0.0, 1.0e-10);  /* cbrt(1e-30) = 1e-10 */
    }
    else {
        float c = exp2(log2(a) * 0.3333333432674408);
        c = c - (c * c * c - a) / (3.0 * c * c);
        R = Interval(c * 0.9999990463256836, c * 1.0000009536743164);  /* 1 -+ 2^-20 */
        if (!(cube_up(R.a) <= a && cube_down(R.b) >= a)) {
            R = Interval(c * 0.999755859375, c * 1.000244140625);  /* 1 -+ 2^-12 */
            if (!(cube_up(R.a) <= a && cube_down(R.b) >= a)) {
                R = Interval(0.0, f_max(a, 1.0));
            }
        }
    }
    return (x < 0.0 ? -R : R);
}

/* Reduction of x to r = x - k pi/2 with |r| <= 0.8 (a bit more than pi/4), with Cody and Waite's
   splitting of pi/2 into PIO2_1 + PIO2_2 + PIO2_3, where PIO2_1 has 8 significant bits so that k*PIO2_1
   is exact for |k| < 2^16. dr bounds |x - k pi/2 - r|. */
static const float TWO_OVER_PI = 0.6366197466850281;
static const float PIO2_1 = 1.5703125;
static const float PIO2_2 = 4.8382679233327508e-4;
static const float PIO2_3 = 2.5633440682570896e-12;
static const float PIO2_ERR = 1.0e-19;  /* |pi/2 - PIO2_1 - PIO2_2 - PIO2_3| < 8.4e-20 */
static const float TRIG_MAX = 51000.0;  /* |k| < 2^15 */

struct TrigReduction {
    float r;
    float dr;
    int quadrant;
    bool ok;
};

TrigReduction trig_reduce(float x) {
    TrigReduction red;
    red.ok = (f_abs(x) <= TRIG_MAX);
    if (!red.ok) { x = 0.0; }
    const float k = round(x * TWO_OVER_PI);
    if (k == 0.0) {
        red.r = x; red.dr = 0.0; red.quadrant = 0;
        return red;
    }
    const float r1 = x - k * PIO2_1;
    const float t2 = k * PIO2_2;
    const float r2 = r1 - t2;
    const float t3 = k * PIO2_3;
    red.r = r2 - t3;
    red.dr = (f_abs(r1) + f_abs(t2) + f_abs(r2) + f_abs(t3) + f_abs(red.r)) * 1.1920928955078125e-7
           + (f_abs(k) * PIO2_ERR + 4.70197740328915e-38);
    red.quadrant = int(k) & 3;
    red.ok = red.ok && (f_abs(red.r) <= 0.8);
    return red;
}

/* Taylor polynomials of sin and cos on |r| <= 0.8, with rounded coefficients. The truncation errors are
   below |r|^11/11! < 2^-28 |r| and r^12/12! < 2^-32, the rounding errors of the evaluation below 2 u |r|
   and 3 u, the bounds |r| 2^-21 and 2^-21 leave a margin of four. */
static const float S3 = -1.6666667163e-01;
static const float S5 = 8.3333337680e-03;
static const float S7 = -1.9841270114e-04;
static const float S9 = 2.7557318845e-06;
static const float C4 = 4.1666667908e-02;
static const float C6 = -1.3888889225e-03;
static const float C8 = 2.4801587642e-05;
static const float C10 = -2.7557319981e-07;
static const float POLY_ERR = 4.76837158203125e-7;  /* 2^-21 */

float sin_poly(float r, float z) {
    float q = S9;
    q = q * z + S7;
    q = q * z + S5;
    q = q * z + S3;
    return r + (r * z) * q;
}

float cos_poly(float z) {
    float q = C10;
    q = q * z + C8;
    q = q * z + C6;
    q = q * z + C4;
    return (1.0 - 0.5 * z) + (z * z) * q;
}

/* sin(x) or cos(x) (shift = 1) with rigorous bounds, [-1, 1] for |x| > 51000 */
Interval sincos_enclosure(float x, int shift) {
    const TrigReduction red = trig_reduce(x);
    if (!red.ok) { return Interval(-1.0, 1.0); }
    const float z = red.r * red.r;
    const int q = (red.quadrant + shift) & 3;
    float v, e;
    if ((q & 1) == 0) {
        v = sin_poly(red.r, z);
        e = f_abs(red.r) * POLY_ERR;
    }
    else {
        v = cos_poly(z);
        e = POLY_ERR;
    }
    if (q >= 2) { v = -v; }
    return intersect(enclose_abs(v, e + red.dr), Interval(-1.0, 1.0));
}

public Interval sin_enclosure(float x) { return sincos_enclosure(x, 0); }
public Interval cos_enclosure(float x) { return sincos_enclosure(x, 1); }

public Interval tan_enclosure(float x) { return sin_enclosure(x) / cos_enclosure(x); }
public Interval cot_enclosure(float x) { return cos_enclosure(x) / sin_enclosure(x); }

/* 1 / cbrt(x) */
public Interval rcbrt_enclosure(float x) { return point(1.0) / cbrt_enclosure(x); }

/* bounds of log2(10), log10(2), ln(10) and 1/ln(10) */
static const float LOG2_10_LO = 3.321928024291992;
static const float LOG2_10_HI = 3.3219282627105713;
static const float LOG10_2_LO = 0.3010299801826477;
static const float LOG10_2_HI = 0.3010300099849701;
static const float LN10_LO = 2.3025848865509033;
static const float LN10_HI = 2.3025851249694824;
static const float INV_LN10_LO = 0.4342944622039795;
static const float INV_LN10_HI = 0.4342944920063019;
static const float EXP10_MAX = 17.0;  /* 10^17 < 2^57 */

/* 10^x = 2^(x log2(10)), exp2 is increasing */
public Interval exp10_enclosure(float x) {
    if (x > EXP10_MAX) { return Interval::entire(); }
    const Interval T = point(x) * Interval(LOG2_10_LO, LOG2_10_HI);
    return Interval(exp2_enclosure(T.a).a, exp2_enclosure(T.b).b);
}

/* log10(x) = log2(x) log10(2), x > 0 is expected */
public Interval log10_enclosure(float x) {
    const Interval L = log2_enclosure(x);
    if (!L.is_bounded()) { return Interval(-HUGE, -30.0); }  /* log10(2^-100) < -30.1 */
    return L * Interval(LOG10_2_LO, LOG10_2_HI);
}

/* x - 2n where 2n is the even integer nearest to x, exact, in [-1, 1]. Floats beyond 2^24 are even. */
float reduce_period2(float x) {
    if (!(f_abs(x) < 16777216.0)) { return 0.0; }
    return x - 2.0 * f_rint(x * 0.5);
}

/* sin(pi r) and cos(pi r) for 0 <= r <= 1/4, where both functions are monotonic. r = 0 may be a flushed
   denormal, |sin(pi r)| is then below 2^-124 and cos(pi r) is in [1 - 2^-24, 1]. */
Interval sinpi_quarter(float r) {
    if (r == 0.0) { return Interval(-UP_ABS, UP_ABS); }
    const Interval A = point(r) * Interval(PI_LO, PI_HI);
    return Interval(f_max(sin_enclosure(A.a).a, 0.0), f_min(sin_enclosure(A.b).b, 1.0));
}

Interval cospi_quarter(float r) {
    if (r == 0.0) { return Interval(0.99999994039535522, 1.0); }
    const Interval A = point(r) * Interval(PI_LO, PI_HI);
    return Interval(f_max(cos_enclosure(A.b).a, 0.0), f_min(cos_enclosure(f_max(A.a, 0.0)).b, 1.0));
}

/* sin(pi x) and cos(pi x), exact reduction to [0, 1/4] */
public Interval sinpi_enclosure(float x) {
    float r = reduce_period2(x);
    const bool negative = (r < 0.0);
    r = f_abs(r);
    if (r > 0.5) { r = 1.0 - r; }  /* sinpi(r) = sinpi(1 - r), exact */
    const Interval S = (r <= 0.25 ? sinpi_quarter(r) : cospi_quarter(0.5 - r));
    return (negative ? -S : S);
}

public Interval cospi_enclosure(float x) {
    float r = f_abs(reduce_period2(x));
    bool negative = false;
    if (r > 0.5) { r = 1.0 - r; negative = true; }  /* cospi(r) = -cospi(1 - r), exact */
    const Interval C = (r <= 0.25 ? cospi_quarter(r) : sinpi_quarter(0.5 - r));
    return (negative ? -C : C);
}


/*** function descriptors: f and f' (approximations), F and DF (rigorous enclosures at a point) ***/

interface IFn {
    float f(float x);
    float df(float x);
    Interval F(float x);
    Interval DF(float x);
    bool has_dfinv();
    /* the point u in [p, q] where f'(u) = d (an approximation), when has_dfinv() */
    float dfinv(float d, float p, float q);
};

/* multiples of a half period are regular points, inflection points or poles */
static const int REGULAR = 0;
static const int INFLECTION = 1;
static const int POLE = 2;

interface ITrigFn : IFn {
    Interval half_period();
    /* kind of the point m * half_period */
    int kind(int m);
    /* curvature on (m * half_period, (m+1) * half_period), +1 convex, -1 concave */
    int curvature(int m);
    /* lipschitz constant of bounded functions */
    float lipschitz();
};

struct ExpFn : IFn {
    float f(float x) { return exp(x); }
    float df(float x) { return exp(x); }
    Interval F(float x) { return exp_enclosure(x); }
    Interval DF(float x) { return exp_enclosure(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-30 ? log(d) : p); }
};

struct Exp2Fn : IFn {
    float f(float x) { return exp2(x); }
    float df(float x) { return 0.6931471805599453 * exp2(x); }
    Interval F(float x) { return exp2_enclosure(x); }
    Interval DF(float x) { return Interval(0.693147122859954833984375, 0.693147182464599609375) * exp2_enclosure(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-30 ? log2(d * 1.4426950408889634) : p); }
};

struct LogFn : IFn {
    float f(float x) { return log(x); }
    float df(float x) { return 1.0 / x; }
    Interval F(float x) { return log_enclosure(x); }
    Interval DF(float x) { return point(1.0) / point(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-30 ? 1.0 / d : q); }
};

struct Log2Fn : IFn {
    float f(float x) { return log2(x); }
    float df(float x) { return 1.4426950408889634 / x; }
    Interval F(float x) { return log2_enclosure(x); }
    Interval DF(float x) { return Interval(1.44269502162933349609375, 1.4426950216293334961 * 1.0000002) / point(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-30 ? 1.4426950408889634 / d : q); }
};

struct SqrtFn : IFn {
    float f(float x) { return sqrt(f_max(x, 0.0)); }
    float df(float x) { return 0.5 / sqrt(f_max(x, 1.0e-30)); }
    Interval F(float x) { return Interval(sqrt_down(x), sqrt_up(x)); }
    Interval DF(float x) { return point(0.5) / F(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-15 ? 0.25 / (d * d) : q); }
};

struct RsqrtFn : IFn {
    float f(float x) { return 1.0 / sqrt(x); }
    float df(float x) { return -0.5 / (x * sqrt(x)); }
    Interval F(float x) { return point(1.0) / Interval(sqrt_down(x), sqrt_up(x)); }
    Interval DF(float x) { return -(point(0.5) * F(x) / point(x)); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
};

struct ReciprocalFn : IFn {
    float f(float x) { return 1.0 / x; }
    float df(float x) { return -1.0 / (x * x); }
    Interval F(float x) { return point(1.0) / point(x); }
    Interval DF(float x) { return -sqr(F(x)); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) {
        return (-d > 1.0e-30 ? (p < 0.0 ? -1.0 : 1.0) / sqrt(-d) : p);
    }
};

/* x^n for an integer n */
struct PownFn : IFn {
    int n;
    float f(float x) { return fpown(x, n); }
    float df(float x) { return float(n) * fpown(x, n - 1); }
    Interval F(float x) { return pow_interval(x, n); }
    Interval DF(float x) { return point(float(n)) * pow_interval(x, n - 1); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    static Interval pow_interval(float x, int k) {
        return (k >= 0 ? ipow(point(x), k) : point(1.0) / ipow(point(x), -k));
    }
};

/* x^y for a real y, x >= 0 */
struct PowFn : IFn {
    float y;
    float f(float x) { return exp2(y * log2(x)); }
    float df(float x) { return y * exp2((y - 1.0) * log2(x)); }
    Interval F(float x) { return pow_enclosure(x, y); }
    Interval DF(float x) {
        if (x == 0.0) { return (y > 1.0 ? point(0.0) : Interval::entire()); }
        return point(y) * F(x) / point(x);
    }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
};

struct CbrtFn : IFn {
    float f(float x) { return (x < 0.0 ? -1.0 : 1.0) * exp2(log2(f_abs(x)) * 0.3333333432674408); }
    float df(float x) { const float c = f(x); return 1.0 / (3.0 * c * c); }
    Interval F(float x) { return cbrt_enclosure(x); }
    Interval DF(float x) { return point(1.0) / (point(3.0) * sqr(F(x))); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
};

struct Exp10Fn : IFn {
    float f(float x) { return exp2(x * 3.3219280948873622); }
    float df(float x) { return 2.302585092994046 * f(x); }
    Interval F(float x) { return exp10_enclosure(x); }
    Interval DF(float x) { return Interval(LN10_LO, LN10_HI) * exp10_enclosure(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-30 ? log2(d * 0.43429448190325176) * 0.3010299956639812 : p); }
};

struct Log10Fn : IFn {
    float f(float x) { return log2(x) * 0.3010299956639812; }
    float df(float x) { return 0.43429448190325176 / x; }
    Interval F(float x) { return log10_enclosure(x); }
    Interval DF(float x) { return Interval(INV_LN10_LO, INV_LN10_HI) / point(x); }
    bool has_dfinv() { return true; }
    float dfinv(float d, float p, float q) { return (d > 1.0e-30 ? 0.43429448190325176 / d : q); }
};

struct RcbrtFn : IFn {
    float f(float x) { return 1.0 / CbrtFn().f(x); }
    float df(float x) { return -1.0 / (3.0 * x * CbrtFn().f(x)); }
    Interval F(float x) { return rcbrt_enclosure(x); }
    Interval DF(float x) { return -(F(x) / (point(3.0) * point(x))); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
};

struct AtanFn : IFn {
    float f(float x) { return atan(x); }
    float df(float x) { return 1.0 / (1.0 + x * x); }
    Interval F(float x) { return atan_enclosure(x); }
    Interval DF(float x) { return point(1.0) / (point(1.0) + sqr(point(x))); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
};

/* half period pi/2 */
Interval half_pi() { return Interval(0.5 * PI_LO, 0.5 * PI_HI); }

struct SinFn : ITrigFn {
    float f(float x) { return sin(x); }
    float df(float x) { return cos(x); }
    Interval F(float x) { return sin_enclosure(x); }
    Interval DF(float x) { return cos_enclosure(x); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    Interval half_period() { return half_pi(); }
    /* sin(m pi/2) = 0 for even m */
    int kind(int m) { return (pmod(m, 2) == 0 ? INFLECTION : REGULAR); }
    /* sin'' = -sin */
    int curvature(int m) { return (pmod(m, 4) < 2 ? -1 : +1); }
    float lipschitz() { return 1.0; }
};

struct CosFn : ITrigFn {
    float f(float x) { return cos(x); }
    float df(float x) { return -sin(x); }
    Interval F(float x) { return cos_enclosure(x); }
    Interval DF(float x) { return -sin_enclosure(x); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    Interval half_period() { return half_pi(); }
    int kind(int m) { return (pmod(m, 2) == 1 ? INFLECTION : REGULAR); }
    /* cos > 0 on (-pi/2, pi/2), cos'' = -cos */
    int curvature(int m) { return (pmod(m, 4) == 0 || pmod(m, 4) == 3 ? -1 : +1); }
    float lipschitz() { return 1.0; }
};

struct TanFn : ITrigFn {
    float f(float x) { return tan(x); }
    float df(float x) { const float t = tan(x); return 1.0 + t * t; }
    Interval F(float x) { return tan_enclosure(x); }
    Interval DF(float x) { return point(1.0) + sqr(F(x)); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    Interval half_period() { return half_pi(); }
    int kind(int m) { return (pmod(m, 2) == 0 ? INFLECTION : POLE); }
    /* tan > 0 on (0, pi/2), tan'' = 2 tan (1 + tan^2) */
    int curvature(int m) { return (pmod(m, 2) == 0 ? +1 : -1); }
    float lipschitz() { return HUGE; }
};

struct CotFn : ITrigFn {
    float f(float x) { return cos(x) / sin(x); }
    float df(float x) { const float c = f(x); return -1.0 - c * c; }
    Interval F(float x) { return cot_enclosure(x); }
    Interval DF(float x) { return -(point(1.0) + sqr(F(x))); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    Interval half_period() { return half_pi(); }
    int kind(int m) { return (pmod(m, 2) == 1 ? INFLECTION : POLE); }
    /* cot > 0 on (0, pi/2), cot'' = 2 cot (1 + cot^2) */
    int curvature(int m) { return (pmod(m, 2) == 0 ? +1 : -1); }
    float lipschitz() { return HUGE; }
};

/* sin(pi x) and cos(pi x), half period 1/2 */
struct SinpiFn : ITrigFn {
    float f(float x) { const Interval S = sinpi_enclosure(x); return S.mid(); }
    float df(float x) { const Interval C = cospi_enclosure(x); return 3.1415927 * C.mid(); }
    Interval F(float x) { return sinpi_enclosure(x); }
    Interval DF(float x) { return Interval(PI_LO, PI_HI) * cospi_enclosure(x); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    Interval half_period() { return point(0.5); }
    int kind(int m) { return SinFn().kind(m); }
    int curvature(int m) { return SinFn().curvature(m); }
    float lipschitz() { return PI_HI; }
};

struct CospiFn : ITrigFn {
    float f(float x) { const Interval C = cospi_enclosure(x); return C.mid(); }
    float df(float x) { const Interval S = sinpi_enclosure(x); return -3.1415927 * S.mid(); }
    Interval F(float x) { return cospi_enclosure(x); }
    Interval DF(float x) { return -(Interval(PI_LO, PI_HI) * sinpi_enclosure(x)); }
    bool has_dfinv() { return false; }
    float dfinv(float d, float p, float q) { return p; }
    Interval half_period() { return point(0.5); }
    int kind(int m) { return CosFn().kind(m); }
    int curvature(int m) { return CosFn().curvature(m); }
    float lipschitz() { return PI_HI; }
};


/*** residuals ***/

/* slope of the secant of f over X, an approximation. For a degenerate interval the slope does not
   matter because all symbol magnitudes are zero, zero yields the tightest enclosure. */
float secant_slope<Fn : IFn>(Fn fn, Interval X) {
    if (!(X.b - X.a > 1.0e-30)) { return 0.0; }
    const float s = (fn.f(X.b) - fn.f(X.a)) / (X.b - X.a);
    return (f_abs(s) <= LIMIT ? s : 0.0);
}

/* approximation of the point u in [p,q] where f'(u) = alpha, the tangent at u bounds g */
float tangent_point<Fn : IFn>(Fn fn, float alpha, float p, float q, int curvature) {
    float u;
    if (fn.has_dfinv()) {
        u = fn.dfinv(alpha, p, q);
    }
    else {
        /* f' is increasing (convex) or decreasing (concave) on [p,q] */
        float lo = p;
        float hi = q;
        for (int i = 0; i < 16; ++i) {
            const float m = lo + (hi - lo) * 0.5;
            if ((curvature > 0) == (fn.df(m) < alpha)) { lo = m; } else { hi = m; }
        }
        u = lo + (hi - lo) * 0.5;
    }
    return f_min(f_max(u, p), q);
}

/* Enclosure of g(x) = f(x) - alpha*x over [p,q].
   curvature: +1 if f is convex on [p,q], -1 if concave, 0 if f is monotonic on [p,q]. */
Interval residual_piece<Fn : IFn>(Fn fn, float alpha, float p, float q, int curvature) {
    const Interval A = point(alpha);
    const Interval Fp = fn.F(p);
    const Interval Fq = fn.F(q);

    if (curvature == 0) {
        return join(Fp, Fq) - A * Interval(p, q);
    }

    /* convex: g is maximal at an endpoint, f(x) >= f(u) + f'(u)(x-u) for all x
       concave: g is minimal at an endpoint, f(x) <= f(u) + f'(u)(x-u) for all x */
    const Interval gp = Fp - A * point(p);
    const Interval gq = Fq - A * point(q);
    const float u = tangent_point(fn, alpha, p, q, curvature);
    const Interval Ui = point(u);
    /* g(x) >= f(u) - alpha*u + (f'(u) - alpha)*(x - u) for convex f (<= for concave f) */
    const Interval t = (fn.F(u) - A * Ui) + (fn.DF(u) - A) * (Interval(p, q) - Ui);
    if (curvature > 0) {
        return Interval(t.a, f_max(gp.b, gq.b));
    }
    else {
        return Interval(f_min(gp.a, gq.a), t.b);
    }
}

/* enclosure of g over X when the curvature of f changes sign at 0 only */
Interval residual_split0<Fn : IFn>(Fn fn, float alpha, Interval X, int curvature_neg, int curvature_pos) {
    if (X.b <= 0.0) { return residual_piece(fn, alpha, X.a, X.b, curvature_neg); }
    if (X.a >= 0.0) { return residual_piece(fn, alpha, X.a, X.b, curvature_pos); }
    return join(residual_piece(fn, alpha, X.a, 0.0, curvature_neg),
                residual_piece(fn, alpha, 0.0, X.b, curvature_pos));
}

/* Enclosure of g over X for trigonometric functions whose special points (inflection points and poles)
   are multiples of a half period. Returns false if X contains a pole or is too large. */
bool residual_periodic<Fn : ITrigFn>(Fn fn, float alpha, Interval X, out Interval g) {
    g = Interval::entire();
    const Interval half_period = fn.half_period();
    const float mf = floor(X.a / half_period.a) - 1.0;
    if (!(f_abs(mf) < 1.0e6)) { return false; }
    const int m0 = int(mf);

    bool first = true;
    float cursor = X.a;
    for (int i = 0; i < 8; ++i) {
        const int m = m0 + i;
        const Interval S = point(float(m)) * half_period;
        if (i == 0) {
            if (!(S.b < X.a)) { return false; }
            continue;
        }
        if (S.b < cursor) {
            continue;
        }
        if (S.a > X.b) {
            const Interval piece = residual_piece(fn, alpha, cursor, X.b, fn.curvature(m - 1));
            g = (first ? piece : join(g, piece));
            return true;
        }
        const int kind = fn.kind(m);
        if (kind == REGULAR) {
            continue;
        }
        if (kind == POLE || !(S.b - S.a < 0.125)) {
            return false;
        }
        /* inflection point, f is monotonic around S */
        if (S.a > cursor) {
            const Interval piece = residual_piece(fn, alpha, cursor, S.a, fn.curvature(m - 1));
            g = (first ? piece : join(g, piece));
            first = false;
        }
        const Interval sliver = residual_piece(fn, alpha, f_max(S.a, cursor), f_min(S.b, X.b), 0);
        g = (first ? sliver : join(g, sliver));
        first = false;
        cursor = S.b;
        if (cursor >= X.b) {
            return true;
        }
    }
    return false;
}

/* single convex (+1) or concave (-1) piece with a known lipschitz constant */
Approx approx_piece<Fn : IFn>(Fn fn, Interval X, int curvature, float lip, float err) {
    const float alpha = secant_slope(fn, X);
    const Interval g = residual_piece(fn, alpha, X.a, X.b, curvature);
    return from_residual(alpha, g, propagate(lip, err));
}

/* convex (+1) or concave (-1) pieces on both sides of 0 */
Approx approx_split0<Fn : IFn>(Fn fn, Interval X, int curvature_neg, int curvature_pos, float prop) {
    const float alpha = secant_slope(fn, X);
    const Interval g = residual_split0(fn, alpha, X, curvature_neg, curvature_pos);
    return from_residual(alpha, g, prop);
}


/*** approximations, X is the range of the affine part and err the input error term ***/

public Approx approx_exp(Interval X, float err) {
    const float xmax = add_up(X.b, err);
    if (!X.is_bounded() || !(xmax <= EXP_MAX)) { return Approx::make_entire(); }
    const ExpFn fn;
    return approx_piece(fn, X, +1, fn.F(xmax).b, err);
}

public Approx approx_exp2(Interval X, float err) {
    const float xmax = add_up(X.b, err);
    if (!X.is_bounded() || !(xmax <= EXP2_MAX)) { return Approx::make_entire(); }
    const Exp2Fn fn;
    return approx_piece(fn, X, +1, fn.DF(xmax).b, err);
}

public Approx approx_exp10(Interval X, float err) {
    const float xmax = add_up(X.b, err);
    if (!X.is_bounded() || !(xmax <= EXP10_MAX)) { return Approx::make_entire(); }
    const Exp10Fn fn;
    return approx_piece(fn, X, +1, fn.DF(xmax).b, err);
}

/* lower bound of the inputs of logarithms, reciprocals and negative powers */
static const float DOMAIN_MIN = 3.469446951953614e-18;  /* 2^-58 */

Approx approx_logarithm<Fn : IFn>(Fn fn, Interval X, float err) {
    if (!X.is_bounded() || !(X.a >= DOMAIN_MIN)) { return Approx::make_entire(); }
    const float xmin = sub_down(X.a, err);
    const float lip = (xmin >= DOMAIN_MIN ? fn.DF(xmin).b : HUGE);
    return approx_piece(fn, X, -1, lip, err);
}

public Approx approx_log(Interval X, float err) { return approx_logarithm(LogFn(), X, err); }
public Approx approx_log2(Interval X, float err) { return approx_logarithm(Log2Fn(), X, err); }
public Approx approx_log10(Interval X, float err) { return approx_logarithm(Log10Fn(), X, err); }

/* the input is restricted to x >= 0 */
public Approx approx_sqrt(Interval Xin, float err) {
    if (!Xin.is_bounded() || Xin.b < 0.0) { return Approx::make_entire(); }
    const Interval X = Interval(f_max(Xin.a, 0.0), Xin.b);
    /* |sqrt(x+r) - sqrt(x)| <= sqrt(|r|) */
    const float holder = (err == 0.0 ? 0.0 : sqrt_up(err));
    if (X.b == 0.0) {
        return Approx::make_affine(0.0, 0.0, holder);
    }
    const SqrtFn fn;
    const float xmin = sub_down(X.a, err);
    const float lip = (xmin > LOG_SMALL ? div_up(1.0, mul_down(2.0, sqrt_down(xmin))) : HUGE);
    return approx_split0(fn, X, -1, -1, f_min(propagate(lip, err), holder));
}

public Approx approx_rsqrt(Interval X, float err) {
    if (!X.is_bounded() || !(X.a >= DOMAIN_MIN)) { return Approx::make_entire(); }
    const RsqrtFn fn;
    const float xmin = sub_down(X.a, err);
    const float lip = (xmin >= DOMAIN_MIN ? f_abs(fn.DF(xmin).a) : HUGE);
    return approx_piece(fn, X, +1, lip, err);
}

/* min-range approximation */
public Approx approx_reciprocal(Interval X, float err) {
    if (!X.is_bounded() || !(X.a >= DOMAIN_MIN || X.b <= -DOMAIN_MIN)) { return Approx::make_entire(); }
    const ReciprocalFn fn;
    const bool positive = (X.a > 0.0);
    const float e = (positive ? X.b : X.a);
    const float alpha = (X.a == X.b ? 0.0 : -1.0 / (e * e));
    const Interval g = residual_piece(fn, alpha, X.a, X.b, positive ? +1 : -1);
    const Interval Xf = widen(X, err);
    float lip = HUGE;
    if (Xf.a >= DOMAIN_MIN || Xf.b <= -DOMAIN_MIN) {
        const float m = Xf.mig();
        lip = div_up(1.0, mul_down(m, m));
    }
    return from_residual(alpha, g, propagate(lip, err));
}

public Approx approx_pown(Interval X, float err, int n) {
    if (n == 0) { return Approx::make_affine(0.0, 1.0, 0.0); }
    if (n == 1) { return Approx::make_affine(1.0, 0.0, err); }
    if (!X.is_bounded()) { return Approx::make_entire(); }
    const Interval Xf = widen(X, err);
    /* the result and its derivative must stay below 2^55 */
    if (n > 0) {
        const float M = Xf.mag();
        if (M > 1.0 && float(n) * log2(M) > 55.0) { return Approx::make_entire(); }
    }
    else {
        const float m = Xf.mig();
        if (!(m >= DOMAIN_MIN) || (m < 1.0 && float(1 - n) * log2(m) < -55.0)) { return Approx::make_entire(); }
    }

    PownFn fn;
    fn.n = n;
    const bool even = (n % 2 == 0);
    const float alpha = secant_slope(fn, X);
    Interval g;
    if (X.a >= 0.0) {
        g = residual_piece(fn, alpha, X.a, X.b, +1);
    }
    else if (X.b <= 0.0) {
        g = residual_piece(fn, alpha, X.a, X.b, even ? +1 : -1);
    }
    else {
        g = (even ? residual_piece(fn, alpha, X.a, X.b, +1) : residual_split0(fn, alpha, X, -1, +1));
    }

    float lip;
    if (n > 0) {
        lip = (point(float(n)) * PownFn::pow_interval(Xf.mag(), n - 1)).b;
    }
    else {
        lip = (point(float(-n)) * PownFn::pow_interval(Xf.mig(), n - 1)).b;
    }
    return from_residual(alpha, g, propagate(lip, err));
}

/* the input is restricted to x >= 0 (x > 0 for negative exponents) */
public Approx approx_pow(Interval Xin, float err, float y) {
    if (!is_bounded(y)) { return Approx::make_entire(); }
    if (y == 0.0) { return Approx::make_affine(0.0, 1.0, 0.0); }
    if (y == 1.0) { return Approx::make_affine(1.0, 0.0, err); }
    if (!Xin.is_bounded() || Xin.b < 0.0 || (y < 0.0 && !(Xin.a >= DOMAIN_MIN))) { return Approx::make_entire(); }

    PowFn fn;
    fn.y = y;
    const Interval X = Interval(f_max(Xin.a, 0.0), Xin.b);
    const Interval Xf = widen(X, err);
    float prop = HUGE;
    if (y > 1.0) {
        prop = propagate(fn.DF(Xf.mag()).b, err);
    }
    else if (y > 0.0) {
        /* |x^y - z^y| <= |x - z|^y */
        const float holder = (err == 0.0 ? 0.0 : fn.F(err).b);
        const float lip = (Xf.a >= DOMAIN_MIN ? fn.DF(Xf.a).b : HUGE);
        prop = f_min(propagate(lip, err), holder);
    }
    else if (Xf.a >= DOMAIN_MIN) {
        prop = propagate(-fn.DF(Xf.a).a, err);
    }

    if (X.b == 0.0) {
        return Approx::make_affine(0.0, 0.0, prop);
    }
    const float alpha = secant_slope(fn, X);
    const Interval g = residual_piece(fn, alpha, X.a, X.b, (y > 1.0 || y < 0.0) ? +1 : -1);
    return from_residual(alpha, g, prop);
}

public Approx approx_cbrt(Interval X, float err) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    const CbrtFn fn;
    /* |cbrt(x+r) - cbrt(x)| <= 2 cbrt(|r|/2) */
    const float holder = (err == 0.0 ? 0.0 : mul_up(2.0, fn.F(mul_up(err, 0.5)).b));
    const Interval Xf = widen(X, err);
    const float m = Xf.mig();
    const float lip = (m >= DOMAIN_MIN ? fn.DF(m).b : HUGE);
    return approx_split0(fn, X, +1, -1, f_min(propagate(lip, err), holder));
}

/* the input must not contain [-2^-58, 2^-58] */
public Approx approx_rcbrt(Interval X, float err) {
    if (!X.is_bounded() || !(X.a >= DOMAIN_MIN || X.b <= -DOMAIN_MIN)) { return Approx::make_entire(); }
    const RcbrtFn fn;
    const Interval Xf = widen(X, err);
    float lip = HUGE;
    if (Xf.a >= DOMAIN_MIN || Xf.b <= -DOMAIN_MIN) {
        /* |f'| is decreasing with |x| */
        lip = fn.DF(Xf.mig()).mag();
    }
    return approx_piece(fn, X, (X.a > 0.0 ? +1 : -1), lip, err);
}

public Approx approx_atan(Interval X, float err) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    return approx_split0(AtanFn(), X, +1, -1, propagate(1.0, err));
}

Approx approx_bounded_trigo<Fn : ITrigFn>(Fn fn, Interval X, float err) {
    /* the whole range [-1,1], the input error does not matter */
    const Approx full = Approx::make_range(-1.0, 1.0);
    const float period = 4.0 * fn.half_period().a;
    if (!X.is_bounded() || !(sub_up(add_up(X.b, err), sub_down(X.a, err)) < period)) {
        return full;
    }
    const float alpha = secant_slope(fn, X);
    Interval g;
    if (!residual_periodic(fn, alpha, X, g)) {
        return full;
    }
    return from_residual(alpha, g, propagate(fn.lipschitz(), err));
}

public Approx approx_sin(Interval X, float err) { return approx_bounded_trigo(SinFn(), X, err); }
public Approx approx_cos(Interval X, float err) { return approx_bounded_trigo(CosFn(), X, err); }
public Approx approx_sinpi(Interval X, float err) { return approx_bounded_trigo(SinpiFn(), X, err); }
public Approx approx_cospi(Interval X, float err) { return approx_bounded_trigo(CospiFn(), X, err); }

/* tan and cot, entire if the input contains a pole */
Approx approx_unbounded_trigo<Fn : ITrigFn>(Fn fn, Interval X, float err) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    const Interval Xf = widen(X, err);
    Interval g;
    /* the widened input must not contain any pole, the derivative is then maximal at an endpoint */
    if (!Xf.is_bounded() || !residual_periodic(fn, 0.0, Xf, g)) {
        return Approx::make_entire();
    }
    const float lip = f_max(fn.DF(Xf.a).mag(), fn.DF(Xf.b).mag());
    const float alpha = secant_slope(fn, X);
    if (!residual_periodic(fn, alpha, X, g)) {
        return Approx::make_entire();
    }
    return from_residual(alpha, g, propagate(lip, err));
}

public Approx approx_tan(Interval X, float err) { return approx_unbounded_trigo(TanFn(), X, err); }
public Approx approx_cot(Interval X, float err) { return approx_unbounded_trigo(CotFn(), X, err); }

/* convex piecewise linear functions f with f(0) = 0, f(x) = sa*x for x < 0 and f(x) = sb*x for x > 0 */
Approx approx_kink(Interval X, float err, float sa, float sb) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    const float prop = propagate(f_max(f_abs(sa), f_abs(sb)), err);
    if (X.a >= 0.0) { return Approx::make_affine(sb, 0.0, prop); }
    if (X.b <= 0.0) { return Approx::make_affine(sa, 0.0, prop); }
    /* g(x) = f(x) - alpha*x is minimal at 0 for sa <= alpha <= sb */
    float alpha = (sb * X.b - sa * X.a) / (X.b - X.a);
    alpha = f_min(f_max(alpha, sa), sb);
    const Interval A = point(alpha);
    const Interval ga = (point(sa) - A) * point(X.a);
    const Interval gb = (point(sb) - A) * point(X.b);
    return from_residual(alpha, Interval(0.0, f_max(ga.b, gb.b)), prop);
}

public Approx approx_abs(Interval X, float err) { return approx_kink(X, err, -1.0, 1.0); }
public Approx approx_positive_part(Interval X, float err) { return approx_kink(X, err, 0.0, 1.0); }


/*** discontinuous functions, X is the range of the whole input (affine part and error term) ***/

/* rounding modes of the integer valued functions */
public static const int TO_FLOOR = 0;
public static const int TO_CEIL = 1;
public static const int TO_TRUNC = 2;
public static const int TO_ROUND = 3;  /* ties away from zero */
public static const int TO_RINT = 4;   /* ties to even */

public float to_integer(int mode, float x) {
    switch (mode) {
    case TO_FLOOR: return floor(x);
    case TO_CEIL: return ceil(x);
    case TO_TRUNC: return trunc(x);
    case TO_ROUND: return f_round(x);
    default: return f_rint(x);
    }
}

/* non-decreasing integer valued functions, exact (and constant) when X does not cross a discontinuity */
public Approx approx_step(Interval X, int mode) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    X = outward_normal(X);
    return Approx::make_range(to_integer(mode, X.a), to_integer(mode, X.b));
}

/* x - floor(x), exactly x - n when floor(x) = n is constant over X */
public Approx approx_fract(Interval X, float err) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    X = outward_normal(X);
    const float n = floor(X.a);
    if (n == floor(X.b)) { return Approx::make_affine(1.0, -n, err); }
    return Approx::make_range(0.0, 1.0);
}

/* x - trunc(x), exactly x - n when trunc(x) = n is constant over X, it has the sign of x */
public Approx approx_modf(Interval X, float err) {
    if (!X.is_bounded()) { return Approx::make_entire(); }
    X = outward_normal(X);
    const float n = trunc(X.a);
    if (n == trunc(X.b)) { return Approx::make_affine(1.0, -n, err); }
    return Approx::make_range(X.a >= 0.0 ? 0.0 : -1.0, X.b <= 0.0 ? 0.0 : 1.0);
}

/* Range of x - n*c for all x in X, where |c| <= m: the remainder of fmod (n = trunc(x/c)) has the sign
   of x and is smaller than |c|, the remainder of remainder (n = rint(x/c)) is at most |c|/2 in magnitude. */
public Interval remainder_range(int mode, Interval X, float m) {
    if (mode == TO_TRUNC) { return Interval(X.a >= 0.0 ? 0.0 : -m, X.b <= 0.0 ? 0.0 : m); }
    const float h = mul_up(m, 0.5);
    return Interval(-h, h);
}

/* range of the quotient n = mode(x/y) for x in X and y in Y, entire if unknown */
public Interval quotient_range(int mode, Interval X, Interval Y) {
    if (!X.is_bounded() || !Y.is_bounded()) { return Interval::entire(); }
    const Interval Q = outward_normal(X / Y);
    if (!Q.is_bounded()) { return Interval::entire(); }
    return Interval(to_integer(mode, Q.a), to_integer(mode, Q.b));
}

/* x - n*c with n = mode(x/c) (trunc for fmod, rint for remainder), exactly x - n*c when n is constant
   over X. Q is the range of n. */
public Approx approx_reduction(Interval X, float err, float c, int mode, Interval Q) {
    if (!Q.is_bounded() || !is_bounded(c)) { return Approx::make_entire(); }
    if (Q.a == Q.b) {
        const float n = Q.a;
        const float p = n * c;
        if (!is_bounded(p)) { return Approx::make_entire(); }
        Approx r;
        r.alpha = 1.0;
        r.zeta = -p;
        r.delta = f_max(sub_up(mul_up(n, c), p), sub_up(p, mul_down(n, c)));
        r.err = err;
        r.entire = false;
        return r;
    }
    const Interval R = remainder_range(mode, X, f_abs(c));
    return Approx::make_range(R.a, R.b);
}

/* bounds of an integer valued range as ints, saturated to the int range */
public int integer_lo(float q) { return (q >= -2147483648.0 && q < 2147483648.0 ? int(q) : -2147483647 - 1); }
public int integer_hi(float q) { return (q >= -2147483648.0 && q < 2147483648.0 ? int(q) : 2147483647); }
