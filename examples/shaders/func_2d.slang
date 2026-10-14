/* this file is part of example 01_2d.py */
import vusvna;

typealias AA2 = AA<2>;

uniform float xmin;
uniform float xmax;
uniform float ymin;
uniform float ymax;
uniform uint level;
RWStructuredBuffer<uint> sign;

public T f<T : IReal>(T x, T y) {
    const T xy = x * y;
    return x * x + y * y + xy - xy * xy * 0.5 - 0.25;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void func_2d(uint3 tid : SV_DispatchThreadID) {
    const uint i = tid.y;
    const uint j = tid.x;
    const uint nsplit = 1u << level;
    if (i >= nsplit || j >= nsplit) { return; }

    const float dx = (xmax - xmin) / float(nsplit);
    const float dy = (ymax - ymin) / float(nsplit);
    const AA2 y = AA2::variable(0, Interval(ymin + float(i) * dy, ymin + float(i + 1) * dy));
    const AA2 x = AA2::variable(1, Interval(xmin + float(j) * dx, xmin + float(j + 1) * dx));

    sign[i * nsplit + j] = uint(f(x, y).sign());
}
