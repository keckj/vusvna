/* Kernels of the tests: evaluation of the operations of vusvna on batches of inputs. */
import vusvna;

/* operation codes, see OPS in test_vusvna.py */
T evaluate<T : IReal>(int op, T x, T y) {
    switch (op) {
    case 0: return x + y;
    case 1: return x - y;
    case 2: return x * y;
    case 3: return x / y;
    case 4: return sqr(x);
    case 5: return abs(x);
    case 6: return sqrt(x);
    case 7: return rsqrt(x);
    case 8: return cbrt(x);
    case 9: return exp(x);
    case 10: return exp2(x);
    case 11: return log(x);
    case 12: return log2(x);
    case 13: return sin(x);
    case 14: return cos(x);
    case 15: return tan(x);
    case 16: return atan(x);
    case 17: return pown(x, 3);
    case 18: return pown(x, -2);
    case 19: return pown(x, 4);
    case 20: return pow(x, 1.5);
    case 21: return pow(x, 0.3);
    case 22: return pow(x, -0.7);
    case 23: return positive_part(x);
    case 24: return min(x, y);
    case 25: return max(x, y);
    case 26: return x * 2.5 + 1.0;
    case 27: return 1.0 / x;
    case 28: return smooth_min(x, y, 0.5);
    case 29: return x - x;
    case 30: return sqr(x) + sqr(y) + x * y - sqr(x * y) * 0.5 - 0.25;
    case 31: return sin(x) * cos(y) + exp(x * y * 0.1);
    case 32: return sqrt(sqr(x) - x + 0.5) / sqrt(sqr(x) + 0.5);
    case 33: return x / 3.0;
    case 34: return 2.0 - x;
    case 35: return hypot(x, y);
    case 36: return floor(x);
    case 37: return ceil(x);
    case 38: return trunc(x);
    case 39: return round(x);
    case 40: return rint(x);
    case 41: return fract(x);
    case 42: { T ip; return modf(x, ip); }
    case 43: return fmod(x, 0.7);
    case 44: return remainder(x, 0.7);
    case 45: return fmod(x, y);
    case 46: return remainder(x, y);
    case 47: return atan2(x, y);
    case 48: return cot(x);
    case 49: return sinpi(x);
    case 50: return cospi(x);
    case 51: return exp10(x);
    case 52: return log10(x);
    case 53: return rcbrt(x);
    case 54: return x - floor(x) - fract(x);
    case 55: { T ip; const T f = modf(x, ip); return f + ip; }
    default: return x;
    }
}

/* inputs: [a, b] and [c, d] are the ranges of the variables, e is added to both as an error term */
struct Input {
    float a, b, c, d, e;
};

uniform int op;
uniform uint count;
StructuredBuffer<Input> inputs;
/* lower and upper bounds of the result, float evaluation at (a, c) */
RWStructuredBuffer<float> outputs;

AA<2> make_input(int k, float lo, float hi, float e) {
    AA<2> v = AA<2>::variable(k, Interval(lo, hi));
    return (e > 0.0 ? v + AA<2>::from_interval(Interval(-e, e)) : v);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void eval_aa(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count) { return; }
    const Input in = inputs[i];
    const AA<2> X = make_input(0, in.a, in.b, in.e);
    const AA<2> Y = make_input(1, in.c, in.d, in.e);
    const AA<2> R = evaluate(op, X, Y);
    const Interval I = R.interval();
    outputs[3 * i + 0] = I.a;
    outputs[3 * i + 1] = I.b;
    outputs[3 * i + 2] = evaluate(op, Real(in.a), Real(in.c)).v;
}

/* rigorous enclosures at points */
[shader("compute")]
[numthreads(64, 1, 1)]
void eval_point(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count) { return; }
    const float x = inputs[i].a;
    const float y = inputs[i].c;
    Interval I;
    switch (op) {
    case 0: I = sin_enclosure(x); break;
    case 1: I = cos_enclosure(x); break;
    case 2: I = tan_enclosure(x); break;
    case 3: I = exp_enclosure(x); break;
    case 4: I = exp2_enclosure(x); break;
    case 5: I = log_enclosure(x); break;
    case 6: I = log2_enclosure(x); break;
    case 7: I = atan_enclosure(x); break;
    case 8: I = cbrt_enclosure(x); break;
    case 9: I = pow_enclosure(x, y); break;
    case 10: I = Interval(sqrt_down(x), sqrt_up(x)); break;
    case 11: I = Interval(div_down(x, y), div_up(x, y)); break;
    case 12: I = Interval(mul_down(x, y), mul_up(x, y)); break;
    case 13: I = Interval(add_down(x, y), add_up(x, y)); break;
    case 14: I = sinpi_enclosure(x); break;
    case 15: I = cospi_enclosure(x); break;
    case 16: I = exp10_enclosure(x); break;
    case 17: I = log10_enclosure(x); break;
    case 18: I = cot_enclosure(x); break;
    case 19: I = rcbrt_enclosure(x); break;
    case 20: I = point(f_round(x)); break;
    case 21: I = point(f_rint(x)); break;
    default: I = Interval::entire(); break;
    }
    outputs[3 * i + 0] = I.a;
    outputs[3 * i + 1] = I.b;
    outputs[3 * i + 2] = 0.0;
}

/* range of the quotient of remquo(x, c) (c = e) or remquo(x, y) (c = 0) */
[shader("compute")]
[numthreads(64, 1, 1)]
void eval_quotient(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count) { return; }
    const Input in = inputs[i];
    const AA<2> X = AA<2>::variable(0, Interval(in.a, in.b));
    int lo, hi;
    if (in.e != 0.0) {
        remquo(X, in.e, lo, hi);
    }
    else {
        remquo(X, AA<2>::variable(1, Interval(in.c, in.d)), lo, hi);
    }
    outputs[3 * i + 0] = float(lo);
    outputs[3 * i + 1] = float(hi);
    outputs[3 * i + 2] = 0.0;
}
