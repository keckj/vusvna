/* this file is part of examples 02_quadtree.py and 03_octree.py */
module compaction;

/* Stream compaction: returns the offset where the calling thread writes its n outputs. Offsets inside a
   wave come from a prefix sum, each wave reserves its range with one atomic operation (the order of the
   waves does not matter). All the threads of the wave must call this function. */
public uint compact(uint n, RWStructuredBuffer<uint> counter) {
    const uint prefix = WavePrefixSum(n);
    const uint total = WaveActiveSum(n);
    uint base = 0;
    if (WaveIsFirstLane()) {
        InterlockedAdd(counter[0], total, base);
    }
    return WaveReadLaneFirst(base) + prefix;
}

/* linear index of a thread of a 2D dispatch of rows of row_width threads */
public uint linear_index(uint3 tid, uint row_width) { return tid.y * row_width + tid.x; }
