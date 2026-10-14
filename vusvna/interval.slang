/* Intervals with outward rounding. Intervals with an endpoint beyond LIMIT are unbounded and their
   arithmetic returns entire(), the interval [-HUGE, HUGE]. */
implementing vusvna;

public enum Sign {
    UNKNOWN = 0,
    POSITIVE = 1,
    NEGATIVE = 2,
};

public struct Interval {
    public float a;
    public float b;

    public __init(float a, float b) { this.a = a; this.b = b; }
    public __init(float x) { a = x; b = x; }

    public static Interval entire() { return Interval(-HUGE, HUGE); }

    public bool is_bounded() { return f_abs(a) <= LIMIT && f_abs(b) <= LIMIT; }
    public bool is_empty() { return a > b; }
    public bool contains(float x) { return a <= x && x <= b; }
    public bool subset(Interval o) { return o.a <= a && b <= o.b; }
    public float mid() { return 0.5 * a + 0.5 * b; }
    /* upper bound of the magnitude */
    public float mag() { return f_max(f_abs(a), f_abs(b)); }
    /* lower bound of the magnitude */
    public float mig() { return (a <= 0.0 && b >= 0.0) ? 0.0 : f_min(f_abs(a), f_abs(b)); }

    public Sign sign() {
        if (a > 0.0) { return Sign::POSITIVE; }
        if (b < 0.0) { return Sign::NEGATIVE; }
        return Sign::UNKNOWN;
    }
};

public Interval point(float x) { return Interval(x, x); }
public Interval join(Interval x, Interval y) { return Interval(f_min(x.a, y.a), f_max(x.b, y.b)); }
public Interval intersect(Interval x, Interval y) { return Interval(f_max(x.a, y.a), f_min(x.b, y.b)); }

public __prefix Interval operator-(Interval x) { return Interval(-x.b, -x.a); }

public Interval operator+(Interval x, Interval y) {
    if (!x.is_bounded() || !y.is_bounded()) { return Interval::entire(); }
    return Interval(add_down(x.a, y.a), add_up(x.b, y.b));
}

public Interval operator-(Interval x, Interval y) {
    if (!x.is_bounded() || !y.is_bounded()) { return Interval::entire(); }
    return Interval(sub_down(x.a, y.b), sub_up(x.b, y.a));
}

public Interval operator*(Interval x, Interval y) {
    if (!x.is_bounded() || !y.is_bounded()) { return Interval::entire(); }
    const float lo = f_min(f_min(mul_down(x.a, y.a), mul_down(x.a, y.b)), f_min(mul_down(x.b, y.a), mul_down(x.b, y.b)));
    const float hi = f_max(f_max(mul_up(x.a, y.a), mul_up(x.a, y.b)), f_max(mul_up(x.b, y.a), mul_up(x.b, y.b)));
    return Interval(lo, hi);
}

/* the divisor must not contain [-2^-60, 2^-60] */
public Interval operator/(Interval x, Interval y) {
    if (!x.is_bounded() || !y.is_bounded() || !(y.a >= 8.673617379884035e-19 || y.b <= -8.673617379884035e-19)) {
        return Interval::entire();
    }
    const float lo = f_min(f_min(div_down(x.a, y.a), div_down(x.a, y.b)), f_min(div_down(x.b, y.a), div_down(x.b, y.b)));
    const float hi = f_max(f_max(div_up(x.a, y.a), div_up(x.a, y.b)), f_max(div_up(x.b, y.a), div_up(x.b, y.b)));
    return Interval(lo, hi);
}

public Interval sqr(Interval x) {
    if (!x.is_bounded()) { return Interval::entire(); }
    if (x.a >= 0.0) { return Interval(mul_down(x.a, x.a), mul_up(x.b, x.b)); }
    if (x.b <= 0.0) { return Interval(mul_down(x.b, x.b), mul_up(x.a, x.a)); }
    return Interval(0.0, f_max(mul_up(x.a, x.a), mul_up(x.b, x.b)));
}

/* square root of the part of x in the domain, x.b >= 0 is expected */
public Interval sqrt(Interval x) {
    if (!x.is_bounded()) { return Interval::entire(); }
    return Interval(sqrt_down(f_max(x.a, 0.0)), sqrt_up(f_max(x.b, 0.0)));
}

public Interval abs(Interval x) {
    if (x.a >= 0.0) { return x; }
    if (x.b <= 0.0) { return -x; }
    return Interval(0.0, f_max(-x.a, x.b));
}

/* the interval [r - e, r + e] where e bounds the error of r */
public Interval enclose_abs(float r, float e) { return Interval(down_abs(r, e), up_abs(r, e)); }

/* the interval of values within k ulps of r */
public Interval enclose_ulps(float r, float k) { return Interval(down_ulps(r, k), up_ulps(r, k)); }
