#ifndef SEGMENT_SIZE
#error missing SEGMENT_SIZE
#endif

#ifndef ITERS
#error missing ITERS
#endif

#ifndef OUTSTRIDE
#error missing OUTSTRIDE
#endif

const uint INTERVAL = ITERS * SEGMENT_SIZE;
const uint M = INTERVAL / 2;
const uint SHARDS = 256;
const uint POLYS_PER_SHARD = (1 << (AFACS - 1)) / SHARDS;
const uint BUCKETS_PER_POLY = ITERS * SEGMENT_SIZE / SUBSEGMENT_SIZE;

// Huge primes are always more than 2^14
const uint HUGE_LOG_OFFSET = 14;
