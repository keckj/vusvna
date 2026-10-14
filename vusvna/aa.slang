/* Affine arithmetic with N noise symbols.
 *
 * An affine form x = b + sum_i x_i e_i + e, where the e_i are shared noise symbols in [-1, 1] and
 * |e| <= err, represents all the values taken by an expression when its inputs vary in their intervals.
 * Each input variable gets its own symbol (variable), the symbols track the linear correlations between
 * intermediate results. Rounding errors and the errors of nonlinear approximations are accumulated in
 * the independent term err (reduced affine arithmetic, AF1): the number of symbols stays N, at the cost
 * of losing the correlations of these errors.
 *
 * Forms whose magnitude |b| + sum |x_i| + err exceeds LIMIT = 2^60 are entire: they represent any real
 * number, and every operation on an entire form returns an entire form.
 */
implementing vusvna;

/* coefficients below the smallest normal number are set to zero, the driver may flush them anyway */
float flush_denormal(float v) { return (f_abs(v) < MIN_NORMAL ? 0.0 : v); }

public struct AA<let N : int> : IReal {
    /* central value */
    public float b;
    /* coefficients of the noise symbols */
    public float x[N];
    /* bound of the independent error term */
    public float err;

    public __init(float c) {
        b = 0.0;
        [ForceUnroll] for (int i = 0; i < N; ++i) { x[i] = 0.0; }
        err = 0.0;
        if (!is_bounded(c)) {
            err = HUGE;
        }
        else if (f_abs(c) < MIN_NORMAL) {
            err = (c != 0.0 ? MIN_NORMAL : 0.0);
        }
        else {
            b = c;
        }
    }

    public static AA<N> entire() { return AA<N>(HUGE); }

    public static AA<N> from_float(float c) { return AA<N>(c); }

    /* the input variable of symbol k (0 <= k < N) that takes all the values of I */
    public static AA<N> variable(int k, Interval I) {
        if (!I.is_bounded() || I.is_empty()) { return entire(); }
        AA<N> r = AA<N>(0.0);
        r.b = flush_denormal(I.mid());
        const float rad = (I.a == I.b ? 0.0 : f_max(sub_up(r.b, I.a), sub_up(I.b, r.b)));
        [ForceUnroll] for (int i = 0; i < N; ++i) { r.x[i] = (i == k ? rad : 0.0); }
        return r;
    }

    /* a quantity that takes any value in I, uncorrelated with the other quantities */
    public static AA<N> from_interval(Interval I) {
        if (!I.is_bounded() || I.is_empty()) { return entire(); }
        AA<N> r = AA<N>(0.0);
        r.b = flush_denormal(I.mid());
        r.err = (I.a == I.b ? 0.0 : f_max(sub_up(r.b, I.a), sub_up(I.b, r.b)));
        return r;
    }

    public bool is_entire() { return !(err <= LIMIT); }

    /* sum of the magnitudes of the coefficients */
    public float radius() {
        float r = 0.0;
        [ForceUnroll] for (int i = 0; i < N; ++i) { r += f_abs(x[i]); }
        return r;
    }

    /* range of the affine part b + sum_i x_i e_i */
    public Interval affine_range() {
        if (is_entire()) { return Interval::entire(); }
        const float r = inflate(radius(), N);
        return Interval(sub_down(b, r), add_up(b, r));
    }

    /* enclosure of all the values of the form */
    public Interval interval() {
        if (is_entire()) { return Interval::entire(); }
        const float r = inflate(radius() + err, N);
        return Interval(sub_down(b, r), add_up(b, r));
    }

    public Sign sign() { return interval().sign(); }

    /* magnitude of the dependency on symbol k */
    public float magnitude(int k) { return f_abs(x[k]); }

    /* Last step of every operation: raw bounds the independent error term of the exact operation and
       the rounding errors made by the computation of r, up to the relative and absolute errors covered
       by inflate. Denormal coefficients are flushed, which is also covered by inflate. */
    static AA<N> finalize(AA<N> r, float raw) {
        r.b = flush_denormal(r.b);
        float s = f_abs(r.b);
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = flush_denormal(r.x[i]);
            s += f_abs(r.x[i]);
        }
        r.err = inflate(raw, N);
        if (!(s + r.err <= LIMIT)) { return entire(); }
        return r;
    }

    /* image of the form by an affine approximation of a function */
    public AA<N> apply(Approx A) {
        if (A.entire || is_entire()) { return entire(); }
        AA<N> r;
        const float ab = A.alpha * b;
        r.b = ab + A.zeta;
        float s = f_abs(ab) + f_abs(r.b);
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = A.alpha * x[i];
            s += f_abs(r.x[i]);
        }
        return finalize(r, A.delta + A.err + U * s);
    }

    /*** IReal ***/

    public AA<N> op_neg() {
        AA<N> r = this;
        r.b = -b;
        [ForceUnroll] for (int i = 0; i < N; ++i) { r.x[i] = -x[i]; }
        return r;
    }

    public AA<N> op_add(AA<N> o) {
        if (is_entire() || o.is_entire()) { return entire(); }
        AA<N> r;
        r.b = b + o.b;
        float s = f_abs(r.b);
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = x[i] + o.x[i];
            s += f_abs(r.x[i]);
        }
        return finalize(r, err + o.err + U * s);
    }

    public AA<N> op_sub(AA<N> o) { return op_add(o.op_neg()); }

    public AA<N> op_addf(float c) {
        if (is_entire() || !is_bounded(c)) { return entire(); }
        AA<N> r = this;
        r.b = b + c;
        return finalize(r, err + U * f_abs(r.b));
    }

    /* c - this */
    public AA<N> op_rsubf(float c) { return op_neg().op_addf(c); }

    public AA<N> op_mulf(float c) {
        if (is_entire() || !is_bounded(c)) { return entire(); }
        if (f_abs(c) < MIN_NORMAL) {
            /* |c x| <= 2^-126 2^60 */
            AA<N> r = AA<N>(0.0);
            r.err = 1.3552527156068805e-20;
            return r;
        }
        AA<N> r;
        r.b = c * b;
        float s = f_abs(r.b);
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = c * x[i];
            s += f_abs(r.x[i]);
        }
        return finalize(r, f_abs(c) * err + U * s);
    }

    public AA<N> op_divf(float c) {
        if (is_entire() || !(f_abs(c) >= 8.673617379884035e-19 && f_abs(c) <= LIMIT)) { return entire(); }
        AA<N> r;
        r.b = b / c;
        float s = f_abs(r.b);
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = x[i] / c;
            s += f_abs(r.x[i]);
        }
        /* divisions are within 2.5 ulps, 5 u relative */
        return finalize(r, err / f_abs(c) + 8.0 * U * s);
    }

    public AA<N> op_mul(AA<N> o) {
        if (is_entire() || o.is_entire()) { return entire(); }
        const float rx = radius();
        const float ry = o.radius();
        /* upper bounds of the magnitudes of the nonconstant parts */
        const float Rx = inflate(rx + err, N);
        const float Ry = inflate(ry + o.err, N);
        AA<N> r;
        r.b = b * o.b;
        float s = f_abs(r.b) + f_abs(b) * ry + f_abs(o.b) * rx;
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = b * o.x[i] + o.b * x[i];
            s += f_abs(r.x[i]);
        }
        return finalize(r, f_abs(b) * o.err + f_abs(o.b) * err + Rx * Ry + U * s);
    }

    public AA<N> fn_sqr() {
        if (is_entire()) { return entire(); }
        /* (b + l)^2 = b^2 + 2 b l + l^2 where 0 <= l^2 <= R^2 */
        const float R = inflate(radius() + err, N);
        const float h = 0.5 * mul_up(R, R);
        const float bb = b * b;
        AA<N> r;
        r.b = bb + h;
        float s = bb + f_abs(r.b);
        [ForceUnroll] for (int i = 0; i < N; ++i) {
            r.x[i] = (2.0 * b) * x[i];
            s += f_abs(r.x[i]);
        }
        return finalize(r, 2.0 * f_abs(b) * err + h + U * s);
    }

    public AA<N> fn_reciprocal() { return apply(approx_reciprocal(affine_range(), err)); }
    public AA<N> op_div(AA<N> o) { return op_mul(o.fn_reciprocal()); }
    /* c / this */
    public AA<N> op_rdivf(float c) { return fn_reciprocal().op_mulf(c); }

    public AA<N> fn_abs() { return apply(approx_abs(affine_range(), err)); }
    public AA<N> fn_positive_part() { return apply(approx_positive_part(affine_range(), err)); }
    public AA<N> fn_sqrt() { return apply(approx_sqrt(affine_range(), err)); }
    public AA<N> fn_rsqrt() { return apply(approx_rsqrt(affine_range(), err)); }
    public AA<N> fn_cbrt() { return apply(approx_cbrt(affine_range(), err)); }
    public AA<N> fn_exp() { return apply(approx_exp(affine_range(), err)); }
    public AA<N> fn_exp2() { return apply(approx_exp2(affine_range(), err)); }
    public AA<N> fn_log() { return apply(approx_log(affine_range(), err)); }
    public AA<N> fn_log2() { return apply(approx_log2(affine_range(), err)); }
    public AA<N> fn_sin() { return apply(approx_sin(affine_range(), err)); }
    public AA<N> fn_cos() { return apply(approx_cos(affine_range(), err)); }
    public AA<N> fn_tan() { return apply(approx_tan(affine_range(), err)); }
    public AA<N> fn_atan() { return apply(approx_atan(affine_range(), err)); }

    public AA<N> fn_pown(int n) {
        if (n == 2) { return fn_sqr(); }
        if (n == -1) { return fn_reciprocal(); }
        return apply(approx_pown(affine_range(), err, n));
    }

    public AA<N> fn_pow(float y) {
        if (y == 0.5) { return fn_sqrt(); }
        return apply(approx_pow(affine_range(), err, y));
    }

    public AA<N> fn_cot() { return apply(approx_cot(affine_range(), err)); }
    public AA<N> fn_sinpi() { return apply(approx_sinpi(affine_range(), err)); }
    public AA<N> fn_cospi() { return apply(approx_cospi(affine_range(), err)); }
    public AA<N> fn_exp10() { return apply(approx_exp10(affine_range(), err)); }
    public AA<N> fn_log10() { return apply(approx_log10(affine_range(), err)); }
    public AA<N> fn_rcbrt() { return apply(approx_rcbrt(affine_range(), err)); }

    /* atan2(this, x): atan(y/x) for x > 0, +-pi/2 - atan(x/y) for y > 0 or y < 0, [-pi, pi] otherwise */
    public AA<N> fn_atan2(AA<N> x) {
        if (is_entire() || x.is_entire()) { return entire(); }
        if (x.interval().a > 0.0) { return op_div(x).fn_atan(); }
        const Interval Y = interval();
        const AA<N> half_pi = from_interval(Interval(0.5 * PI_LO, 0.5 * PI_HI));
        if (Y.a > 0.0) { return half_pi.op_sub(x.op_div(this).fn_atan()); }
        if (Y.b < 0.0) { return half_pi.op_neg().op_sub(x.op_div(this).fn_atan()); }
        return from_interval(Interval(-PI_HI, PI_HI));
    }

    /* Discontinuous functions are evaluated on the whole range of their input. They are exact (and
       correlated with their input) when the range does not cross a discontinuity, else the result is the
       range of the function, uncorrelated with the input. */
    public AA<N> fn_floor() { return apply(approx_step(interval(), TO_FLOOR)); }
    public AA<N> fn_ceil() { return apply(approx_step(interval(), TO_CEIL)); }
    public AA<N> fn_trunc() { return apply(approx_step(interval(), TO_TRUNC)); }
    public AA<N> fn_round() { return apply(approx_step(interval(), TO_ROUND)); }
    public AA<N> fn_rint() { return apply(approx_step(interval(), TO_RINT)); }
    public AA<N> fn_fract() { return apply(approx_fract(interval(), err)); }
    public AA<N> fn_modf() { return apply(approx_modf(interval(), err)); }

    /* x - n*c with n = trunc(x/c) (fmod) or n = rint(x/c) (remainder), the range of n is returned */
    AA<N> reduce(float c, int mode, out int quo_lo, out int quo_hi) {
        const Interval X = interval();
        const Interval Q = quotient_range(mode, X, point(c));
        quo_lo = integer_lo(Q.a);
        quo_hi = integer_hi(Q.b);
        return apply(approx_reduction(X, err, c, mode, Q));
    }

    /* the same with an affine divisor, entire if y may be zero */
    AA<N> reduce_by(AA<N> y, int mode, out int quo_lo, out int quo_hi) {
        const Interval X = interval();
        const Interval Y = y.interval();
        const Interval Q = quotient_range(mode, X, Y);
        quo_lo = integer_lo(Q.a);
        quo_hi = integer_hi(Q.b);
        if (!Q.is_bounded()) { return entire(); }
        if (Q.a == Q.b) { return op_sub(y.op_mulf(Q.a)); }
        return from_interval(remainder_range(mode, X, Y.mag()));
    }

    public AA<N> fn_fmod(float c) { int lo, hi; return reduce(c, TO_TRUNC, lo, hi); }
    public AA<N> fn_remquo(float c, out int quo_lo, out int quo_hi) { return reduce(c, TO_RINT, quo_lo, quo_hi); }
    public AA<N> fn_fmod_by(AA<N> y) { int lo, hi; return reduce_by(y, TO_TRUNC, lo, hi); }
    public AA<N> fn_remquo_by(AA<N> y, out int quo_lo, out int quo_hi) { return reduce_by(y, TO_RINT, quo_lo, quo_hi); }
};
