/* Functions built on top of the elementary operations of IReal types. */
implementing vusvna;

/* min(x, y) = (x + y - |x - y|)/2 and max(x, y) = (x + y + |x - y|)/2, so that min(x, y) + max(x, y)
   cancels to x + y. When x - y has a known sign, the result is exactly x or y (up to rounding). */
public __generic<T : IReal> T min(T x, T y) { return ((x + y) - abs(x - y)) * 0.5; }
public __generic<T : IReal> T max(T x, T y) { return ((x + y) + abs(x - y)) * 0.5; }
public __generic<T : IReal> T min(T x, float c) { return ((x + c) - abs(x - c)) * 0.5; }
public __generic<T : IReal> T max(T x, float c) { return ((x + c) + abs(x - c)) * 0.5; }
public __generic<T : IReal> T min(float c, T x) { return min(x, c); }
public __generic<T : IReal> T max(float c, T x) { return max(x, c); }

public __generic<T : IReal> T clamp(T x, float lo, float hi) { return min(max(x, lo), hi); }
public __generic<T : IReal> T clamp(T x, T lo, T hi) { return min(max(x, lo), hi); }

public __generic<T : IReal> T hypot(T x, T y) { return sqrt(sqr(x) + sqr(y)); }
public __generic<T : IReal> T hypot(T x, T y, T z) { return sqrt(sqr(x) + sqr(y) + sqr(z)); }
public __generic<T : IReal> T rhypot(T x, T y) { return rsqrt(sqr(x) + sqr(y)); }
public __generic<T : IReal> T rhypot(T x, T y, T z) { return rsqrt(sqr(x) + sqr(y) + sqr(z)); }

public __generic<T : IReal> T lerp(T a, T b, T t) { return a + t * (b - a); }
public __generic<T : IReal> T lerp(T a, T b, float t) { return a + (b - a) * t; }

/* hermite interpolation t^2 (3 - 2t) with t = clamp((x - e0)/(e1 - e0), 0, 1) */
public __generic<T : IReal> T smoothstep(float e0, float e1, T x) {
    const T t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return sqr(t) * (3.0 - 2.0 * t);
}

/* polynomial smooth minimum of signed distances with a blending radius k > 0:
   min(a, b) - h^2 k/4 with h = max(k - |a - b|, 0)/k */
public __generic<T : IReal> T smooth_min(T a, T b, float k) {
    const T h = positive_part(k - abs(a - b)) / k;
    return min(a, b) - sqr(h) * (k * 0.25);
}

public __generic<T : IReal> T smooth_max(T a, T b, float k) { return -smooth_min(-a, -b, k); }

/* x^y = exp(y log(x)) for x > 0 (entire otherwise) */
public __generic<T : IReal> T pow(T x, T y) { return exp(y * log(x)); }

/* Predicates: certainly_* are true only if the relation holds for all values represented by the forms,
   so !certainly_lt(x, y) does not mean x >= y. possibly_zero(x) is false only if x is certainly nonzero. */
public __generic<T : IReal> bool certainly_lt(T x, T y) { return (x - y).interval().b < 0.0; }
public __generic<T : IReal> bool certainly_le(T x, T y) { return (x - y).interval().b <= 0.0; }
public __generic<T : IReal> bool certainly_gt(T x, T y) { return (x - y).interval().a > 0.0; }
public __generic<T : IReal> bool certainly_ge(T x, T y) { return (x - y).interval().a >= 0.0; }
public __generic<T : IReal> bool certainly_lt(T x, float c) { return x.interval().b < c; }
public __generic<T : IReal> bool certainly_le(T x, float c) { return x.interval().b <= c; }
public __generic<T : IReal> bool certainly_gt(T x, float c) { return x.interval().a > c; }
public __generic<T : IReal> bool certainly_ge(T x, float c) { return x.interval().a >= c; }
public __generic<T : IReal> bool possibly_zero(T x) { return x.interval().contains(0.0); }
