"""Containment tests of vusvna: the enclosures computed on the GPU must contain the values of the
functions computed in double precision at many sample points.

Run with pytest or directly: python tests/test_vusvna.py
"""

import math
import pathlib
import sys

import numpy as np
import slangpy as spy

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIMIT = 2.0**60

_device = None
_kernels = {}


def device():
    global _device
    if _device is None:
        _device = spy.Device(
            type=spy.DeviceType.vulkan,
            compiler_options=spy.SlangCompilerOptions({"include_paths": [str(ROOT / "vusvna")]}),
        )
    return _device


def kernel(entry):
    if entry not in _kernels:
        dev = device()
        program = dev.load_program(str(ROOT / "tests" / "kernels.slang"), [entry])
        _kernels[entry] = dev.create_compute_kernel(program)
    return _kernels[entry]


def run(entry, op, a, b=None, c=None, d=None, e=None):
    """Evaluates op on the inputs, returns the lower bounds, the upper bounds and the float values."""
    n = len(a)
    zeros = np.zeros(n, np.float32)
    data = np.stack([np.asarray(v if v is not None else zeros, np.float32) for v in (a, b, c, d, e)], axis=1)
    dev = device()
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    inputs = dev.create_buffer(element_count=n, struct_size=20, usage=usage, data=np.ascontiguousarray(data))
    outputs = dev.create_buffer(element_count=3 * n, struct_size=4, usage=usage)
    kernel(entry).dispatch(spy.uint3(n, 1, 1), vars={"inputs": inputs, "outputs": outputs, "op": op, "count": n})
    dev.wait()
    out = outputs.to_numpy().view(np.float32).reshape(n, 3).astype(np.float64)
    return out[:, 0], out[:, 1], out[:, 2]


def smooth_min(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


C07 = float(np.float32(0.7))


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _remainder(x, y):
    return math.remainder(x, y) if y != 0.0 else math.nan


ieee_remainder = np.vectorize(_remainder, otypes=[np.float64])


def fold_period2(x):
    """Exact reduction of x to [0, 1/2] for sinpi and cospi: returns (s, sign of sinpi, sign of cospi)."""
    r = x - 2.0 * np.round(x * 0.5)
    ssign = np.where(r < 0.0, -1.0, 1.0)
    r = np.abs(r)
    csign = np.where(r > 0.5, -1.0, 1.0)
    return np.where(r > 0.5, 1.0 - r, r), ssign, csign


def sinpi(x):
    s, ssign, _ = fold_period2(x)
    return ssign * np.where(s <= 0.25, np.sin(np.pi * s), np.cos(np.pi * (0.5 - s)))


def cospi(x):
    s, _, csign = fold_period2(x)
    return csign * np.where(s <= 0.25, np.cos(np.pi * s), np.sin(np.pi * (0.5 - s)))


def nonneg(x):
    return np.maximum(x, 0.0)


def positive(x):
    return np.where(x > 0.0, x, np.nan)


def nonzero(x):
    return np.where(x != 0.0, x, np.nan)


# code: (name, reference, unary). Functions restricted to x >= 0 are evaluated on the part of the input
# in their domain, the others must return an entire enclosure when the input leaves the domain.
OPS = {
    0: ("x + y", lambda x, y: x + y, False),
    1: ("x - y", lambda x, y: x - y, False),
    2: ("x * y", lambda x, y: x * y, False),
    3: ("x / y", lambda x, y: x / nonzero(y), False),
    4: ("sqr", lambda x, y: x * x, True),
    5: ("abs", lambda x, y: np.abs(x), True),
    6: ("sqrt", lambda x, y: np.sqrt(nonneg(x)), True),
    7: ("rsqrt", lambda x, y: 1.0 / np.sqrt(positive(x)), True),
    8: ("cbrt", lambda x, y: np.cbrt(x), True),
    9: ("exp", lambda x, y: np.exp(x), True),
    10: ("exp2", lambda x, y: np.exp2(x), True),
    11: ("log", lambda x, y: np.log(positive(x)), True),
    12: ("log2", lambda x, y: np.log2(positive(x)), True),
    13: ("sin", lambda x, y: np.sin(x), True),
    14: ("cos", lambda x, y: np.cos(x), True),
    15: ("tan", lambda x, y: np.tan(x), True),
    16: ("atan", lambda x, y: np.arctan(x), True),
    17: ("pown(x, 3)", lambda x, y: x**3, True),
    18: ("pown(x, -2)", lambda x, y: 1.0 / nonzero(x) ** 2, True),
    19: ("pown(x, 4)", lambda x, y: x**4, True),
    20: ("pow(x, 1.5)", lambda x, y: nonneg(x) ** 1.5, True),
    21: ("pow(x, 0.3)", lambda x, y: nonneg(x) ** 0.3, True),
    22: ("pow(x, -0.7)", lambda x, y: positive(x) ** -0.7, True),
    23: ("positive_part", lambda x, y: nonneg(x), True),
    24: ("min", lambda x, y: np.minimum(x, y), False),
    25: ("max", lambda x, y: np.maximum(x, y), False),
    26: ("x * 2.5 + 1", lambda x, y: x * 2.5 + 1.0, True),
    27: ("1 / x", lambda x, y: 1.0 / nonzero(x), True),
    28: ("smooth_min", lambda x, y: smooth_min(x, y, 0.5), False),
    29: ("x - x", lambda x, y: x - x, True),
    30: ("x2+y2+xy-(xy)2/2-1/4", lambda x, y: x * x + y * y + x * y - (x * y) ** 2 * 0.5 - 0.25, False),
    31: ("sin(x)cos(y)+exp(xy/10)", lambda x, y: np.sin(x) * np.cos(y) + np.exp(x * y * 0.1), False),
    32: ("g(x) of 00_1d", lambda x, y: np.sqrt(x * x - x + 0.5) / np.sqrt(x * x + 0.5), True),
    33: ("x / 3", lambda x, y: x / 3.0, True),
    34: ("2 - x", lambda x, y: 2.0 - x, True),
    35: ("hypot", lambda x, y: np.hypot(x, y), False),
    36: ("floor", lambda x, y: np.floor(x), True),
    37: ("ceil", lambda x, y: np.ceil(x), True),
    38: ("trunc", lambda x, y: np.trunc(x), True),
    39: ("round", lambda x, y: round_away(x), True),
    40: ("rint", lambda x, y: np.rint(x), True),
    41: ("fract", lambda x, y: x - np.floor(x), True),
    42: ("modf", lambda x, y: x - np.trunc(x), True),
    43: ("fmod(x, 0.7)", lambda x, y: np.fmod(x, C07), True),
    44: ("remainder(x, 0.7)", lambda x, y: ieee_remainder(x, C07), True),
    45: ("fmod(x, y)", lambda x, y: np.fmod(x, nonzero(y)), False),
    46: ("remainder(x, y)", lambda x, y: ieee_remainder(x, y), False),
    47: ("atan2", lambda x, y: np.arctan2(x, y), False),
    48: ("cot", lambda x, y: 1.0 / np.tan(nonzero(x)), True),
    49: ("sinpi", lambda x, y: sinpi(x), True),
    50: ("cospi", lambda x, y: cospi(x), True),
    51: ("exp10", lambda x, y: 10.0**x, True),
    52: ("log10", lambda x, y: np.log10(positive(x)), True),
    53: ("rcbrt", lambda x, y: 1.0 / np.cbrt(nonzero(x)), True),
    54: ("x - floor(x) - fract(x)", lambda x, y: 0.0 * x, True),
    55: ("modf(x) + ip", lambda x, y: x, True),
}


def random_ranges(rng, n):
    """Random input ranges [a, b] with an error term e, over many scales."""
    mag = 10.0 ** rng.uniform(-4, 2.5, n)
    center = rng.choice([-1.0, 1.0], n) * mag
    width = np.where(rng.random(n) < 0.5, mag * 10.0 ** rng.uniform(-7, 0.3, n), 10.0 ** rng.uniform(-6, 1, n))
    width[rng.random(n) < 0.1] = 0.0
    special = rng.random(n)
    center = np.where(special < 0.05, rng.uniform(-1e-30, 1e-30, n), center)
    center = np.where((special >= 0.05) & (special < 0.08), rng.uniform(-1e15, 1e15, n), center)
    a = (center - 0.5 * width).astype(np.float32)
    b = (center + 0.5 * width).astype(np.float32)
    a, b = np.minimum(a, b), np.maximum(a, b)
    e = np.where(rng.random(n) < 0.5, 0.0, (b.astype(np.float64) - a) * 10.0 ** rng.uniform(-3, -0.5, n))
    return a, b, e.astype(np.float32)


def check_op(code, n=4096, samples=33, seed=0):
    name, ref, unary = OPS[code]
    rng = np.random.default_rng(seed + 1000 * code)
    a, b, e = random_ranges(rng, n)
    c, d, f = random_ranges(rng, n)
    lo, hi, _ = run("eval_aa", code, a, b, c, d, np.maximum(e, f) if not unary else e)
    e64 = (np.maximum(e, f) if not unary else e).astype(np.float64)

    t = np.linspace(0.0, 1.0, samples)
    xa, xb = a.astype(np.float64) - e64, b.astype(np.float64) + e64
    ya, yb = c.astype(np.float64) - e64, d.astype(np.float64) + e64
    xs = xa[:, None] + (xb - xa)[:, None] * t[None, :]
    ys = ya[:, None] + (yb - ya)[:, None] * t[None, :]
    if unary:
        values = ref(xs, xs)
    else:
        values = ref(xs[:, :, None], ys[:, None, :]).reshape(n, -1)

    with np.errstate(invalid="ignore"):
        bad_domain = np.isnan(values).any(axis=1)
        vmin = np.nanmin(np.where(np.isnan(values), np.inf, values), axis=1)
        vmax = np.nanmax(np.where(np.isnan(values), -np.inf, values), axis=1)
    unbounded_lo = lo <= -LIMIT
    unbounded_hi = hi >= LIMIT
    entire = unbounded_lo & unbounded_hi
    ok = entire | (~bad_domain & (unbounded_lo | (lo <= vmin)) & (unbounded_hi | (hi >= vmax)))
    assert not np.isnan(lo).any() and not np.isnan(hi).any(), f"{name}: NaN bounds"
    failures = np.flatnonzero(~ok)
    if failures.size:
        i = failures[0]
        raise AssertionError(
            f"{name}: {failures.size} failures, x=[{a[i]!r}, {b[i]!r}] y=[{c[i]!r}, {d[i]!r}] e={e64[i]!r}: "
            f"[{lo[i]!r}, {hi[i]!r}] does not contain [{vmin[i]!r}, {vmax[i]!r}]"
        )

    # tightness: computed width over sampled width for bounded results of nondegenerate inputs
    sel = ~entire & ~bad_domain & (vmax - vmin > 1e-6 * np.maximum(np.abs(vmax), 1e-3))
    ratio = np.median((hi[sel] - lo[sel]) / (vmax[sel] - vmin[sel])) if sel.any() else np.nan
    return {"name": name, "entire": entire.mean(), "ratio": ratio}


POINT_OPS = {
    0: ("sin", lambda x, y: np.sin(x)),
    1: ("cos", lambda x, y: np.cos(x)),
    2: ("tan", lambda x, y: np.tan(x)),
    3: ("exp", lambda x, y: np.exp(x)),
    4: ("exp2", lambda x, y: np.exp2(x)),
    5: ("log", lambda x, y: np.log(x)),
    6: ("log2", lambda x, y: np.log2(x)),
    7: ("atan", lambda x, y: np.arctan(x)),
    8: ("cbrt", lambda x, y: np.cbrt(x)),
    9: ("pow", lambda x, y: x**y),
    10: ("sqrt", lambda x, y: np.sqrt(x)),
    11: ("div", lambda x, y: x / y),
    12: ("mul", lambda x, y: x * y),
    13: ("add", lambda x, y: x + y),
    14: ("sinpi", lambda x, y: sinpi(x)),
    15: ("cospi", lambda x, y: cospi(x)),
    16: ("exp10", lambda x, y: 10.0**x),
    17: ("log10", lambda x, y: np.log10(x)),
    18: ("cot", lambda x, y: 1.0 / np.tan(x)),
    19: ("rcbrt", lambda x, y: 1.0 / np.cbrt(x)),
    20: ("round", lambda x, y: round_away(x)),
    21: ("rint", lambda x, y: np.rint(x)),
}


def point_inputs(code, rng, n):
    sign = rng.choice([-1.0, 1.0], n)
    if code in (0, 1):
        x = sign * 10.0 ** rng.uniform(-8, np.log10(60000.0), n)
        # neighbours of multiples of pi/2, where the reduction cancels
        k = rng.integers(-30000, 30000, n // 4)
        x[: n // 4] = k * (np.pi / 2) * (1.0 + rng.uniform(-1e-6, 1e-6, n // 4))
        return x, None
    if code == 2:
        return sign * 10.0 ** rng.uniform(-8, 4, n), None
    if code == 3:
        return rng.uniform(-110.0, 45.0, n), None
    if code == 4:
        return rng.uniform(-140.0, 62.0, n), None
    if code in (5, 6, 10):
        return 10.0 ** rng.uniform(-37, 25, n), None
    if code in (7, 8):
        return sign * 10.0 ** rng.uniform(-37, 20, n), None
    if code == 9:
        return 10.0 ** rng.uniform(-25, 12, n), rng.uniform(-6.0, 6.0, n)
    if code in (14, 15):
        x = sign * 10.0 ** rng.uniform(-8, 9, n)
        # multiples of 1/4 and their neighbours
        q = rng.integers(-4000000, 4000000, n // 2) * 0.25
        x[: n // 2] = q * (1.0 + rng.choice([0.0, 1e-7, -1e-7], n // 2))
        return x, None
    if code == 16:
        return rng.uniform(-40.0, 18.0, n), None
    if code == 17:
        return 10.0 ** rng.uniform(-37, 25, n), None
    if code == 18:
        return sign * 10.0 ** rng.uniform(-8, 4, n), None
    if code == 19:
        return sign * 10.0 ** rng.uniform(-37, 20, n), None
    if code in (20, 21):
        x = sign * 10.0 ** rng.uniform(-3, 9, n)
        x[: n // 2] = rng.integers(-2000000, 2000000, n // 2) * 0.5  # ties
        return x, None
    y = rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(-17, 17, n)
    return sign * 10.0 ** rng.uniform(-17, 17, n), y


def check_point(code, n=1 << 18, seed=0):
    name, ref = POINT_OPS[code]
    rng = np.random.default_rng(seed + 77 * code)
    x, y = point_inputs(code, rng, n)
    x = x.astype(np.float32)
    y = (y if y is not None else np.zeros(n)).astype(np.float32)
    lo, hi, _ = run("eval_point", code, x, None, y)
    with np.errstate(all="ignore"):
        values = ref(x.astype(np.float64), y.astype(np.float64))
    assert not np.isnan(lo).any() and not np.isnan(hi).any(), f"{name}: NaN bounds"
    ok = ((lo <= values) | (lo <= -LIMIT)) & ((hi >= values) | (hi >= LIMIT))
    failures = np.flatnonzero(~ok)
    if failures.size:
        i = failures[0]
        raise AssertionError(
            f"{name}: {failures.size} failures, x={x[i]!r} y={y[i]!r}: [{lo[i]!r}, {hi[i]!r}] does not contain {values[i]!r}"
        )
    bounded = (lo > -LIMIT) & (hi < LIMIT) & (np.abs(values) > 1e-30)
    rel = np.median((hi[bounded] - lo[bounded]) / np.abs(values[bounded])) if bounded.any() else np.nan
    return {"name": name, "relative_width": rel}


def check_quotient(n=4096, seed=0):
    """The quotient range of remquo must contain rint(x/c) for all x in X."""
    rng = np.random.default_rng(seed + 31)
    a, b, _ = random_ranges(rng, n)
    c, d, _ = random_ranges(rng, n)
    e = np.where(rng.random(n) < 0.5, np.float32(0.7), 0.0).astype(np.float32)
    lo, hi, _ = run("eval_quotient", 0, a, b, c, d, e)
    t = np.linspace(0.0, 1.0, 33)
    xs = a[:, None] + (b.astype(np.float64) - a)[:, None] * t[None, :]
    ys = np.where(e[:, None] != 0.0, e[:, None].astype(np.float64), c[:, None] + (d.astype(np.float64) - c)[:, None] * t[None, :])
    with np.errstate(all="ignore"):
        q = np.rint(xs[:, :, None] / ys[:, None, :]).reshape(n, -1)
        ok = ((lo[:, None] <= q) & (q <= hi[:, None])) | ~np.isfinite(q) | (lo[:, None] <= -2.0**31) | (hi[:, None] >= 2.0**31 - 1)
    failures = np.flatnonzero(~ok.all(axis=1))
    if failures.size:
        i = failures[0]
        raise AssertionError(f"remquo quotient: {failures.size} failures, x=[{a[i]!r}, {b[i]!r}]: [{lo[i]}, {hi[i]}]")
    return {"known": (lo == hi).mean()}


def test_remquo_quotient():
    check_quotient()


def test_operations():
    for code in OPS:
        check_op(code)


def test_point_enclosures():
    for code in POINT_OPS:
        check_point(code)


def test_wide_and_degenerate():
    """Inputs at the limits: huge ranges, points, tiny values."""
    a = np.array([-1e18, -1e30, 0.0, 1e-38, -3.0, 1e-30, 0.0, -1e15], np.float32)
    b = np.array([1e18, 1e30, 0.0, 2e-38, 3.0, 1e-30, 1e-40, 1e15], np.float32)
    for code in OPS:
        lo, hi, _ = run("eval_aa", code, a, b, a, b)
        assert not np.isnan(lo).any() and not np.isnan(hi).any()
        assert (lo <= hi).all(), f"{OPS[code][0]}: empty result"


if __name__ == "__main__":
    failed = False
    for code in OPS:
        try:
            r = check_op(code)
            print(f"{r['name']:>26}: ok  entire {100 * r['entire']:5.1f}%  median width ratio {r['ratio']:.4g}")
        except AssertionError as err:
            failed = True
            print(f"FAILED {err}")
    for code in POINT_OPS:
        try:
            r = check_point(code)
            print(f"{'point ' + r['name']:>26}: ok  median relative width {r['relative_width']:.3g}")
        except AssertionError as err:
            failed = True
            print(f"FAILED {err}")
    try:
        r = check_quotient()
        print(f"{'remquo quotient':>26}: ok  known {100 * r['known']:5.1f}%")
    except AssertionError as err:
        failed = True
        print(f"FAILED {err}")
    try:
        test_wide_and_degenerate()
        print(f"{'wide and degenerate':>26}: ok")
    except AssertionError as err:
        failed = True
        print(f"FAILED {err}")
    sys.exit(1 if failed else 0)
