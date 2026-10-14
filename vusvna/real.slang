/* Generic real numbers.
 *
 * Functions written once for a generic type T : IReal can be evaluated with floats (Real) or with affine
 * forms (AA<N>), for instance a signed distance function used both to shade a surface and to bound it:
 *
 *     T shape<T : IReal>(T x, T y, T z) { return sqr(x) + sqr(y) + sqr(z) - 1.0; }
 *
 * The operators and the functions below apply to any IReal type. They are not defined for float, so that
 * they never shadow the builtin functions: wrap floats into Real to call generic functions on them.
 */
implementing vusvna;

public interface IReal {
    public static This from_float(float c);
    /* enclosure of the values */
    public Interval interval();

    public This op_neg();
    public This op_add(This o);
    public This op_sub(This o);
    public This op_mul(This o);
    public This op_div(This o);
    public This op_addf(float c);
    public This op_rsubf(float c);
    public This op_mulf(float c);
    public This op_divf(float c);
    public This op_rdivf(float c);

    public This fn_sqr();
    public This fn_reciprocal();
    public This fn_abs();
    public This fn_positive_part();
    public This fn_sqrt();
    public This fn_rsqrt();
    public This fn_cbrt();
    public This fn_exp();
    public This fn_exp2();
    public This fn_log();
    public This fn_log2();
    public This fn_sin();
    public This fn_cos();
    public This fn_tan();
    public This fn_atan();
    public This fn_pown(int n);
    public This fn_pow(float y);
    public This fn_cot();
    public This fn_sinpi();
    public This fn_cospi();
    public This fn_exp10();
    public This fn_log10();
    public This fn_rcbrt();
    /* atan2(this, x) */
    public This fn_atan2(This x);

    public This fn_floor();
    public This fn_ceil();
    public This fn_trunc();
    public This fn_round();
    public This fn_rint();
    public This fn_fract();
    /* this - trunc(this) */
    public This fn_modf();
    public This fn_fmod(float c);
    public This fn_remquo(float c, out int quo_lo, out int quo_hi);
    public This fn_fmod_by(This y);
    public This fn_remquo_by(This y, out int quo_lo, out int quo_hi);
};

/* plain float evaluation of generic functions */
public struct Real : IReal {
    public float v;

    public __init(float c) { v = c; }
    public static Real from_float(float c) { return Real(c); }
    public Interval interval() { return point(v); }

    public Real op_neg() { return Real(-v); }
    public Real op_add(Real o) { return Real(v + o.v); }
    public Real op_sub(Real o) { return Real(v - o.v); }
    public Real op_mul(Real o) { return Real(v * o.v); }
    public Real op_div(Real o) { return Real(v / o.v); }
    public Real op_addf(float c) { return Real(v + c); }
    public Real op_rsubf(float c) { return Real(c - v); }
    public Real op_mulf(float c) { return Real(v * c); }
    public Real op_divf(float c) { return Real(v / c); }
    public Real op_rdivf(float c) { return Real(c / v); }

    public Real fn_sqr() { return Real(v * v); }
    public Real fn_reciprocal() { return Real(1.0 / v); }
    public Real fn_abs() { return Real(abs(v)); }
    public Real fn_positive_part() { return Real(max(v, 0.0)); }
    public Real fn_sqrt() { return Real(sqrt(max(v, 0.0))); }
    public Real fn_rsqrt() { return Real(rsqrt(v)); }
    public Real fn_cbrt() { return Real(sign(v) * exp2(log2(abs(v)) * 0.3333333432674408)); }
    public Real fn_exp() { return Real(exp(v)); }
    public Real fn_exp2() { return Real(exp2(v)); }
    public Real fn_log() { return Real(log(v)); }
    public Real fn_log2() { return Real(log2(v)); }
    public Real fn_sin() { return Real(sin(v)); }
    public Real fn_cos() { return Real(cos(v)); }
    public Real fn_tan() { return Real(tan(v)); }
    public Real fn_atan() { return Real(atan(v)); }
    public Real fn_pown(int n) { return Real(fpown(v, n)); }
    public Real fn_pow(float y) { return Real(v > 0.0 ? exp2(y * log2(v)) : 0.0); }
    public Real fn_cot() { return Real(cos(v) / sin(v)); }
    public Real fn_sinpi() { return Real(sin(3.14159265358979 * reduce_period2(v))); }
    public Real fn_cospi() { return Real(cos(3.14159265358979 * reduce_period2(v))); }
    public Real fn_exp10() { return Real(exp2(v * 3.3219280948873622)); }
    public Real fn_log10() { return Real(log2(v) * 0.3010299956639812); }
    public Real fn_rcbrt() { return Real(1.0 / fn_cbrt().v); }
    public Real fn_atan2(Real x) { return Real(atan2(v, x.v)); }

    public Real fn_floor() { return Real(floor(v)); }
    public Real fn_ceil() { return Real(ceil(v)); }
    public Real fn_trunc() { return Real(trunc(v)); }
    public Real fn_round() { return Real(f_round(v)); }
    public Real fn_rint() { return Real(f_rint(v)); }
    public Real fn_fract() { return Real(v - floor(v)); }
    public Real fn_modf() { return Real(v - trunc(v)); }
    public Real fn_fmod(float c) { return Real(v - trunc(v / c) * c); }
    public Real fn_remquo(float c, out int quo_lo, out int quo_hi) {
        const float n = f_rint(v / c);
        quo_lo = integer_lo(n);
        quo_hi = integer_hi(n);
        return Real(v - n * c);
    }
    public Real fn_fmod_by(Real y) { return fn_fmod(y.v); }
    public Real fn_remquo_by(Real y, out int quo_lo, out int quo_hi) { return fn_remquo(y.v, quo_lo, quo_hi); }
};

public __generic<T : IReal> __prefix T operator-(T x) { return x.op_neg(); }
public __generic<T : IReal> T operator+(T x, T y) { return x.op_add(y); }
public __generic<T : IReal> T operator-(T x, T y) { return x.op_sub(y); }
public __generic<T : IReal> T operator*(T x, T y) { return x.op_mul(y); }
public __generic<T : IReal> T operator/(T x, T y) { return x.op_div(y); }
public __generic<T : IReal> T operator+(T x, float c) { return x.op_addf(c); }
public __generic<T : IReal> T operator+(float c, T x) { return x.op_addf(c); }
public __generic<T : IReal> T operator-(T x, float c) { return x.op_addf(-c); }
public __generic<T : IReal> T operator-(float c, T x) { return x.op_rsubf(c); }
public __generic<T : IReal> T operator*(T x, float c) { return x.op_mulf(c); }
public __generic<T : IReal> T operator*(float c, T x) { return x.op_mulf(c); }
public __generic<T : IReal> T operator/(T x, float c) { return x.op_divf(c); }
public __generic<T : IReal> T operator/(float c, T x) { return x.op_rdivf(c); }

public __generic<T : IReal> T sqr(T x) { return x.fn_sqr(); }
public __generic<T : IReal> T reciprocal(T x) { return x.fn_reciprocal(); }
public __generic<T : IReal> T abs(T x) { return x.fn_abs(); }
/* max(x, 0) */
public __generic<T : IReal> T positive_part(T x) { return x.fn_positive_part(); }
public __generic<T : IReal> T sqrt(T x) { return x.fn_sqrt(); }
public __generic<T : IReal> T rsqrt(T x) { return x.fn_rsqrt(); }
public __generic<T : IReal> T cbrt(T x) { return x.fn_cbrt(); }
public __generic<T : IReal> T exp(T x) { return x.fn_exp(); }
public __generic<T : IReal> T exp2(T x) { return x.fn_exp2(); }
public __generic<T : IReal> T log(T x) { return x.fn_log(); }
public __generic<T : IReal> T log2(T x) { return x.fn_log2(); }
public __generic<T : IReal> T sin(T x) { return x.fn_sin(); }
public __generic<T : IReal> T cos(T x) { return x.fn_cos(); }
public __generic<T : IReal> T tan(T x) { return x.fn_tan(); }
public __generic<T : IReal> T atan(T x) { return x.fn_atan(); }
/* x^n for an integer n */
public __generic<T : IReal> T pown(T x, int n) { return x.fn_pown(n); }
/* x^y for x >= 0 */
public __generic<T : IReal> T pow(T x, float y) { return x.fn_pow(y); }

public __generic<T : IReal> T cot(T x) { return x.fn_cot(); }
/* sin(pi x) and cos(pi x) */
public __generic<T : IReal> T sinpi(T x) { return x.fn_sinpi(); }
public __generic<T : IReal> T cospi(T x) { return x.fn_cospi(); }
public __generic<T : IReal> T exp10(T x) { return x.fn_exp10(); }
public __generic<T : IReal> T log10(T x) { return x.fn_log10(); }
/* 1 / cbrt(x) */
public __generic<T : IReal> T rcbrt(T x) { return x.fn_rcbrt(); }
public __generic<T : IReal> T atan2(T y, T x) { return y.fn_atan2(x); }
public __generic<T : IReal> T atan2(T y, float x) { return y.fn_atan2(T::from_float(x)); }
public __generic<T : IReal> T atan2(float y, T x) { return T::from_float(y).fn_atan2(x); }

/* Discontinuous functions: exact when the range of the input does not cross a discontinuity, else the
   range of the result. round rounds ties away from zero, rint to even. */
public __generic<T : IReal> T floor(T x) { return x.fn_floor(); }
public __generic<T : IReal> T ceil(T x) { return x.fn_ceil(); }
public __generic<T : IReal> T trunc(T x) { return x.fn_trunc(); }
public __generic<T : IReal> T round(T x) { return x.fn_round(); }
public __generic<T : IReal> T rint(T x) { return x.fn_rint(); }
/* x - floor(x) */
public __generic<T : IReal> T fract(T x) { return x.fn_fract(); }
/* returns x - trunc(x) and stores trunc(x) in ip */
public __generic<T : IReal> T modf(T x, out T ip) { ip = x.fn_trunc(); return x.fn_modf(); }

/* fmod(x, y) = x - trunc(x/y) y, entire if y may be zero */
public __generic<T : IReal> T fmod(T x, float c) { return x.fn_fmod(c); }
public __generic<T : IReal> T fmod(T x, T y) { return x.fn_fmod_by(y); }
public __generic<T : IReal> T fmod(float c, T y) { return T::from_float(c).fn_fmod_by(y); }

/* remainder(x, y) = x - rint(x/y) y. remquo also returns the range [quo_lo, quo_hi] of the integer
   quotient rint(x/y) (the full quotient, not only its low bits as in C), saturated to the int range:
   quo_lo == quo_hi means that the quotient is known and the remainder is exactly x - quo y. */
public __generic<T : IReal> T remquo(T x, float c, out int quo_lo, out int quo_hi) { return x.fn_remquo(c, quo_lo, quo_hi); }
public __generic<T : IReal> T remquo(T x, T y, out int quo_lo, out int quo_hi) { return x.fn_remquo_by(y, quo_lo, quo_hi); }
public __generic<T : IReal> T remquo(float c, T y, out int quo_lo, out int quo_hi) {
    return T::from_float(c).fn_remquo_by(y, quo_lo, quo_hi);
}
public __generic<T : IReal> T remainder(T x, float c) { int lo, hi; return x.fn_remquo(c, lo, hi); }
public __generic<T : IReal> T remainder(T x, T y) { int lo, hi; return x.fn_remquo_by(y, lo, hi); }
public __generic<T : IReal> T remainder(float c, T y) { int lo, hi; return T::from_float(c).fn_remquo_by(y, lo, hi); }
