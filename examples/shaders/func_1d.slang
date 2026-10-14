/* this file is part of example 00_1d.py */
import vusvna;

typealias AA1 = AA<1>;

uniform float xmin;
uniform float xmax;
uniform uint nintervals;
uniform uint k;
RWStructuredBuffer<float> a;
RWStructuredBuffer<float> b;

T g<T : IReal>(T x) {
    const T xx = x * x;
    return sqrt(xx - x + 0.5) / sqrt(xx + 0.5);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void func_1d(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= nintervals) { return; }

    const float dx = (xmax - xmin) / float(nintervals);
    const AA1 x = AA1::variable(0, Interval(xmin + float(i) * dx, xmin + float(i + 1) * dx));
    const AA1 y = (k == 0 ? g(x) : g(g(x)));

    const Interval yi = y.interval();
    a[i] = yi.a;
    b[i] = yi.b;
}
