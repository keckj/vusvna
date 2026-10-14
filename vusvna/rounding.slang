/* Outward rounding without directed rounding modes, fused multiply-add, infinities nor NaNs.
 *
 * Vulkan only guarantees round to nearest for additions, subtractions and multiplications. Divisions,
 * square roots and transcendental functions have documented error bounds (Vulkan specification,
 * "Precision and Operation of SPIR-V Instructions"), denormals may be flushed to zero on input and
 * output of any instruction, and unless the SignedZeroInfNanPreserve execution mode is enabled (Slang
 * does not emit it) the implementation may assume that no infinity nor NaN is ever produced. Drivers
 * do exploit this: on NVIDIA isinf(+inf) is false and NaN == NaN is true.
 *
 * vusvna is therefore written so that infinities and NaNs never appear: all magnitudes are kept below
 * LIMIT = 2^60, so that products of two magnitudes and sums of products cannot overflow, and quantities
 * above LIMIT mean "anything". Error free transformations (TwoSum, TwoProd) are not used either, since
 * the driver is free to contract a*b + c into a fused multiply-add.
 *
 * Rounding errors are bounded with the standard model: the result r of an operation whose exact result
 * is x satisfies |r - x| <= k ulp(x) + 2^-126, where k = 1/2 for correctly rounded operations, k is the
 * documented bound otherwise, and 2^-126 accounts for the flush of a denormal result. Contracting an
 * operation into a fused multiply-add only removes a rounding, the bounds remain valid.
 */
implementing vusvna;

/* unit roundoff 2^-24 */
public static const float U = 5.9604644775390625e-8;
/* smallest normal number 2^-126 */
public static const float MIN_NORMAL = 1.1754943508222875e-38;
/* all finite magnitudes stay below LIMIT = 2^60, larger values mean unbounded */
public static const float LIMIT = 1.152921504606846976e18;
/* value used for unbounded quantities, 2^100 */
public static const float HUGE = 1.2676506002282294e30;
/* lower and upper bounds of pi */
public static const float PI_LO = 3.1415925025939941406;
public static const float PI_HI = 3.1415927410125732422;

/* relative and absolute margins of up and down: 2^-22 and 2^-124 */
static const float UP_REL = 2.384185791015625e-7;
static const float UP_ABS = 4.70197740328915e-38;

/* Builtin float functions. The generic functions defined by vusvna for IReal types do not apply to
   float, these wrappers only make the intent explicit inside the library. */
float f_abs(float x) { return abs(x); }
float f_min(float x, float y) { return min(x, y); }
float f_max(float x, float y) { return max(x, y); }

/* true if |x| <= LIMIT, false for unbounded values */
public bool is_bounded(float x) { return f_abs(x) <= LIMIT; }

/* Upper and lower bounds of x, where r is the round to nearest result of a correctly rounded operation
   whose exact result is x. The margin covers the rounding of x, the flush of a denormal x, and the
   rounding of the addition of the margin itself. */
public float up(float r) { return r + (f_abs(r) * UP_REL + UP_ABS); }
public float down(float r) { return r - (f_abs(r) * UP_REL + UP_ABS); }

/* Margin of a result r within k ulps of the exact result x, plus the rounding of r + margin.
   ulp(x) <= 2^-23 |x| and |x| <= |r| (1 + k 2^-23), hence |r - x| <= 2k u |r| (1 + k 2^-22). */
float ulps_margin(float r, float k) {
    return f_abs(r) * (((2.0 * k + 4.0) + k * k * 9.5367431640625e-7) * U) + UP_ABS;
}

/* bounds of x where r is within k ulps of x */
public float up_ulps(float r, float k) { return r + ulps_margin(r, k); }
public float down_ulps(float r, float k) { return r - ulps_margin(r, k); }

/* bounds of x where |r - x| <= e (e >= 0) */
public float up_abs(float r, float e) { return up(r + e); }
public float down_abs(float r, float e) { return down(r - e); }

/* Directed operations on floats. Denormal operands flushed to zero by the driver change the result of
   a product by at most 2^-126 times the other operand, hence the (|a| + |b|) 2^-125 term. */
public float add_up(float a, float b) { return up(a + b); }
public float add_down(float a, float b) { return down(a + b); }
public float sub_up(float a, float b) { return up(a - b); }
public float sub_down(float a, float b) { return down(a - b); }
public float mul_up(float a, float b) {
    const float p = a * b;
    return p + (f_abs(p) * UP_REL + ((f_abs(a) + f_abs(b)) * 2.350988701644575e-38 + UP_ABS));
}
public float mul_down(float a, float b) {
    const float p = a * b;
    return p - (f_abs(p) * UP_REL + ((f_abs(a) + f_abs(b)) * 2.350988701644575e-38 + UP_ABS));
}

/* Division is within 2.5 ulps for divisors of magnitude in [2^-126, 2^126]. Callers ensure that
   2^-60 <= |b| <= 2^60, a flushed denormal dividend a changes the quotient by at most 2^-126 / |b|. */
static const float DIV_ULPS = 2.5;
public float div_up(float a, float b) {
    const float q = a / b;
    return up_ulps(q, DIV_ULPS) + 2.350988701644575e-38 / f_abs(b);
}
public float div_down(float a, float b) {
    const float q = a / b;
    return down_ulps(q, DIV_ULPS) - 2.350988701644575e-38 / f_abs(b);
}

/* Square root is inherited from 1 / inversesqrt(x): inversesqrt is within 2 ulps and division within
   2.5 ulps, which gives less than 5 ulps, 8 are used. Small arguments, where inversesqrt may overflow
   or be flushed, are handled separately: sqrt(x) <= 2^-50 for x <= 2^-100. */
static const float SQRT_ULPS = 8.0;
static const float SQRT_SMALL = 7.888609052210118e-31;  /* 2^-100 */
static const float SQRT_SMALL_ROOT = 8.881784197001252e-16;  /* 2^-50 */
public float sqrt_up(float x) {
    if (x <= SQRT_SMALL) { return SQRT_SMALL_ROOT; }
    return up_ulps(sqrt(x), SQRT_ULPS);
}
public float sqrt_down(float x) {
    if (x <= SQRT_SMALL) { return 0.0; }
    return f_max(down_ulps(sqrt(x), SQRT_ULPS), 0.0);
}

/* Sum of nonnegative terms computed with round to nearest, inflated to an upper bound of the exact
   sum of the exact terms. Each term must be within (N + 8) u of an upper bound of the exact term, the
   relative error of the summation of N + 8 terms is below (N + 8) u, the factor 1 + (4N + 32) u covers
   both with a margin, the absolute term covers the flush of up to 64 (N + 8) denormal results. */
public float inflate(float s, int N) {
    return s * (1.0 + float(4 * N + 32) * U) + float(N + 8) * 7.52316384526264e-37;
}
