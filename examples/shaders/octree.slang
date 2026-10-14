/* this file is part of example 03_octree.py */
import vusvna;
import compaction;

typealias AA3 = AA<3>;

uniform float xmin;
uniform float xmax;
uniform float ymin;
uniform float ymax;
uniform float zmin;
uniform float zmax;
uniform uint start_level;
uniform uint current_level;
uniform uint current_active_cells;
uniform uint max_active_cells;
uniform uint row_width;
StructuredBuffer<uint64_t> active_cells_in;
RWStructuredBuffer<uint64_t> active_cells_out;
RWStructuredBuffer<uint> active_cells_counter;

T f<T : IReal>(T x, T y, T z) {
    const T xx = x * x;
    const T yy = y * y;
    const T zz = z * z;
    return (x - 2.0) * (x - 2.0) * (x + 2.0) * (x + 2.0)
         + (y - 2.0) * (y - 2.0) * (y + 2.0) * (y + 2.0)
         + (z - 2.0) * (z - 2.0) * (z + 2.0) * (z + 2.0)
         + (xx * yy + xx * zz + yy * zz)
         + 6.0 * x * y * z
         - 10.0 * (xx + yy + zz)
         + 22.0;
}

/* each active cell of the previous level is split in 8 cells, the cells that may contain a zero of f
   are the active cells of the current level */
[shader("compute")]
[numthreads(256, 1, 1)]
void octree(uint3 tid : SV_DispatchThreadID) {
    const uint t = linear_index(tid, row_width);

    uint nactive = 0;
    uint64_t cid[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    if (t < current_active_cells) {
        const uint64_t oid = (current_level == start_level ? uint64_t(t) : active_cells_in[t]);
        const uint64_t osplit = uint64_t(1) << (current_level - 1);
        const uint64_t old_i = oid / (osplit * osplit);
        const uint64_t old_j = (oid / osplit) % osplit;
        const uint64_t old_k = oid % osplit;

        const uint64_t nsplit = uint64_t(1) << current_level;
        const float dz = (zmax - zmin) / float(nsplit);
        const float dy = (ymax - ymin) / float(nsplit);
        const float dx = (xmax - xmin) / float(nsplit);

        [ForceUnroll]
        for (uint offset = 0; offset < 8; ++offset) {
            const uint64_t i = 2 * old_i + ((offset & 0x4) >> 2);
            const uint64_t j = 2 * old_j + ((offset & 0x2) >> 1);
            const uint64_t k = 2 * old_k + (offset & 0x1);

            const AA3 z = AA3::variable(0, Interval(zmin + float(i) * dz, zmin + float(i + 1) * dz));
            const AA3 y = AA3::variable(1, Interval(ymin + float(j) * dy, ymin + float(j + 1) * dy));
            const AA3 x = AA3::variable(2, Interval(xmin + float(k) * dx, xmin + float(k + 1) * dx));

            if (f(x, y, z).sign() == Sign::UNKNOWN) {
                cid[nactive] = (i * nsplit + j) * nsplit + k;
                nactive++;
            }
        }
    }

    const uint output_index = compact(nactive, active_cells_counter);
    for (uint offset = 0; offset < nactive; ++offset) {
        active_cells_out[min(max_active_cells - 1, output_index + offset)] = cid[offset];
    }
}
