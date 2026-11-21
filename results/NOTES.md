# Results log

## 01 vecadd (RTX 3070)
- n = 2^26: 1.96 ms, ~410 GB/s (92% of peak); predicted 1.96 ms
- Launch-bound below ~64K elements (~1.8 µs floor)
- L2-resident up to 3.75 MiB working set (~1.1 TB/s), cliff at 4 MiB
- DRAM regime fits t ≈ 4.4 µs + bytes / 411 GB/s

## 02 transpose
- naive, write-coalesced (strided reads): ~339 GB/s at 1024² (L2-affected; re-measure at 8192²)
- naive, read-coalesced (strided writes): ~140 GB/s at 1024²

## 03 matmul (1024^3, FP32; full tables in 03_matmul/RESULTS.md)
| Kernel | GFLOP/s | % of cuBLAS |
|---|---|---|
| naive, row from threadIdx.x (16 x 16) | 306 | 2.8% |
| coalesced, col from threadIdx.x (32 x 8) | 1,423 | 12.9% |
| tiled, shared memory (TILE 16) | 1,611 | 14.6% |
| tiled + coarsening (CF 4) | 1,691 | 15.3% |
| 1D block tiling (64x64 tile, 8 results/thread) | 4,294 | 38.8% |
| cuBLAS SGEMM (no TF32) | 11,070 | 100% |

- Naive kernel time is proportional to L1 sectors per load (ncu: 4.5 / 8.5 / 16.5 sectors -> 3.9 / 7.0 / 14.3 ms).
- Coalesced 32 x 8 runs a memory link at 98% of peak: bandwidth-bound, so only data reuse helps.
- 1024-thread blocks cap occupancy at 66.7% (1 block/SM, 1536 threads/SM on sm_86) -> latency-bound.
- Conventions that prevent bugs: x -> col, y -> row, translate once at the top; test odd non-square shapes.
- Register reuse is the lever past tiling: caching one Bs value across 8 FMAs gave 2.7x over tiled.
