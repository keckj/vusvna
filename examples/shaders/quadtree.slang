/* this file is part of example 02_quadtree.py */
import vusvna;
import compaction;

typealias AA2 = AA<2>;

uniform float xmin;
uniform float xmax;
uniform float ymin;
uniform float ymax;
uniform uint start_level;
uniform uint current_level;
uniform uint current_active_cells;
uniform uint max_active_cells;
uniform uint row_width;
StructuredBuffer<uint64_t> active_cells_in;
RWStructuredBuffer<uint64_t> active_cells_out;
RWStructuredBuffer<uint> active_cells_counter;

T f<T : IReal>(T x, T y) {
    const T xy = x * y;
    return x * x + y * y + xy - xy * xy * 0.5 - 0.25;
}

/* each active cell of the previous level is split in 4 cells, the cells that may contain a zero of f
   are the active cells of the current level */
[shader("compute")]
[numthreads(256, 1, 1)]
void quadtree(uint3 tid : SV_DispatchThreadID) {
    const uint t = linear_index(tid, row_width);

    uint nactive = 0;
    uint64_t cid[4] = { 0, 0, 0, 0 };

    if (t < current_active_cells) {
        const uint64_t oid = (current_level == start_level ? uint64_t(t) : active_cells_in[t]);
        const uint64_t osplit = uint64_t(1) << (current_level - 1);
        const uint64_t old_i = oid / osplit;
        const uint64_t old_j = oid % osplit;

        const uint64_t nsplit = uint64_t(1) << current_level;
        const float dx = (xmax - xmin) / float(nsplit);
        const float dy = (ymax - ymin) / float(nsplit);

        [ForceUnroll]
        for (uint offset = 0; offset < 4; ++offset) {
            const uint64_t i = 2 * old_i + ((offset & 0x2) >> 1);
            const uint64_t j = 2 * old_j + (offset & 0x1);

            const AA2 y = AA2::variable(0, Interval(ymin + float(i) * dy, ymin + float(i + 1) * dy));
            const AA2 x = AA2::variable(1, Interval(xmin + float(j) * dx, xmin + float(j + 1) * dx));

            if (f(x, y).sign() == Sign::UNKNOWN) {
                cid[nactive] = i * nsplit + j;
                nactive++;
            }
        }
    }

    const uint output_index = compact(nactive, active_cells_counter);
    for (uint offset = 0; offset < nactive; ++offset) {
        active_cells_out[min(max_active_cells - 1, output_index + offset)] = cid[offset];
    }
}
