# 03 matmul results (RTX 3070, FP32 peak ~20 TFLOP/s)

## Correctness (non-square, block 16x16)
| Kernel | M x K x N | ms | GFLOP/s | errors |
|---|---|---|---|---|
| naive   | 256 x 512 x 128 | 0.1316 | 255.0 | 0 |
| swapped | 256 x 512 x 128 | 0.0346 | 969.5 | 0 |
| swapped | 128 x 512 x 256 | 0.0345 | 971.4 | 0 |

## Block-shape sweep, 1024 x 1024 x 1024
Note: the sweep loop ran in zsh, which doesn't word-split `$b`, so `by` stayed at its default of 16.
Actual shapes are bx x 16; the three 32 x 16 rows are repeats of the same config.

| Kernel | Block (bx x by) | Warp covers | ms | GFLOP/s |
|---|---|---|---|---|
| naive   |  8 x 16 |  8 rows x 4 cols | 3.383 | 634.7 |
| naive   | 16 x 16 | 16 rows x 2 cols | 7.401 | 290.2 |
| naive   | 32 x 16 | 32 rows x 1 col  | 13.009 / 13.279 / 13.561 | 165.1 / 161.7 / 158.4 |
| naive   | 64 x 16 | 32 rows x 1 col  | 13.497 | 159.1 |
| swapped |  8 x 16 |  8 cols x 4 rows | 2.155 | 996.5 |
| swapped | 16 x 16 | 16 cols x 2 rows | 1.741 | 1233.6 |
| swapped | 32 x 16 | 32 cols x 1 row  | 1.723 (x3) | 1246.1 |
| swapped | 64 x 16 | 32 cols x 1 row  | 1.802 | 1191.6 |

Best: swapped 32 x 16 at 1246 GFLOP/s, ~6% of peak; 7.5x faster than naive 32 x 16, 1.96x faster than best naive (8 x 16).

## Observations
- Naive: time roughly doubles each time bx doubles up to 32 (8 -> 16 -> 32: 3.4 -> 7.4 -> 13.3 ms).
  Each A load in a warp touches one sector per distinct row, and the number of distinct rows per warp is min(bx, 32).
  64 x 16 matches 32 x 16 because a warp can't span more than 32 rows.
- Swapped: block shape barely matters once a warp row is >= 16 floats (full 32-byte sectors for B and C).
  8 x 16 is ~20% slower: 4 distinct A broadcasts per warp and only 1 sector of B per row.
- Run-to-run noise on repeats: ~4% for naive, <0.01% for swapped.

## ncu: global-load sectors per request (1024^3)
Requests = warps x K x 2 loads = 32768 x 1024 x 2 = 67,108,864 (matches exactly).
Prediction per warp-instruction: A load = 1 sector per distinct row (16-byte rows can't share a sector
when strided by K), B load = sectors covered by the warp's distinct consecutive columns.

| Kernel | Block | Requests | Sectors | Sectors/request | Predicted (A, B -> avg) |
|---|---|---|---|---|---|
| naive   | 32 x 8  | 67,108,864 | 1,107,285,993 | **16.50** | 32, 1 -> 16.5 |
| naive   | 16 x 16 | 67,108,864 |   570,424,832 | **8.50**  | 16, 1 -> 8.5  |
| naive   | 8 x 32  | 67,108,864 |   301,989,795 | **4.50**  | 8, 1 -> 4.5   |
| swapped | 32 x 8  | 67,108,864 |   167,772,160 | **2.50**  | 1, 4 -> 2.5   |
| swapped | 16 x 16 | 67,108,864 |   134,217,048 | **2.00**  | 2, 2 -> 2.0   |
| swapped | 8 x 32  | 67,108,864 |   167,760,807 | **2.50**  | 4, 1 -> 2.5   |

All six match the sector model to within 0.01%. (Counts a few thousand below the exact
value come from ncu's multi-pass collection, not from the access pattern.)
Naive sectors/request scale linearly with bx (rows per warp); swapped stays at 2-2.5 for every shape.

Timings under ncu differ from normal runs (6.39 ms vs 7.40 ms) because ncu locks GPU clocks to base by default
(`--clock-control none` disables that); use ncu for counters, not timing.

## Timing at the intended block shapes, 1024^3 (with ncu sectors/request)
| Kernel | Block | Warp covers | Sectors/req | ms | GFLOP/s |
|---|---|---|---|---|---|
| naive   | 32 x 8  | 32 rows x 1 col  | 16.5 | 14.285 | 150.3 |
| naive   | 32 x 4  | 32 rows x 1 col  | —    | 13.980 | 153.6 |
| naive   | 32 x 32 | 32 rows x 1 col  | —    | 13.973 | 153.7 |
| naive   | 16 x 16 | 16 rows x 2 cols | 8.5  | 7.020  | 305.9 |
| naive   | 8 x 32  | 8 rows x 4 cols  | 4.5  | 3.935  | 545.7 |
| swapped | 32 x 8  | 32 cols x 1 row  | 2.5  | **1.509** | **1423.2** |
| swapped | 32 x 4  | 32 cols x 1 row  | —    | 1.717  | 1251.0 |
| swapped | 16 x 16 | 16 cols x 2 rows | 2.0  | 1.740  | 1234.3 |
| swapped | 32 x 32 | 32 cols x 1 row  | —    | 1.895  | 1133.0 |
| swapped | 8 x 32  | 8 cols x 4 rows  | 2.5  | 2.169  | 990.1  |

Best: swapped 32 x 8, 1423 GFLOP/s (~7% of FP32 peak), 9.5x faster than naive at the same block.

### Observations
- Naive: time is proportional to sectors/request — 14.29 / 7.02 / 3.94 ms for 16.5 / 8.5 / 4.5
  sectors (~0.85 ms per sector/request). The kernel is bound by L1 sector throughput on the strided A loads.
  Every 32-wide block shape lands at ~14 ms regardless of by.
- Swapped: sectors/request no longer predicts time. 32 x 8 and 8 x 32 both measure 2.5 sectors/request but
  differ by 44% (1.51 vs 2.17 ms), so a different limit applies once accesses are efficient.
  Hypotheses (unverified):
  - Distinct 128-byte lines per warp load: 32 x 8 = 1 for A + 1 for B; 16 x 16 = 2 + 1; 8 x 32 = 4 + 1.
    Same ordering as the times.
  - 32 x 32 (1024 threads/block): only 1 block fits per SM (sm_86 max 1536 threads/SM), so occupancy is 2/3;
    the 256-thread shapes fit 6 blocks for full occupancy.
  Checked with ncu (below).

### ncu memory + occupancy, swapped kernel, 1024^3
| Block | Mem Busy | Max Bandwidth | L1 hit | L2 hit | Theoretical occ. | Achieved occ. | ms |
|---|---|---|---|---|---|---|---|
| 32 x 8  | 49.2% | **98.2%** | 87.6% | 87.8% | 100%   | 97.4% | 1.51 |
| 8 x 32  | **97.9%** | 78.3% | 87.8% | 89.1% | 100%   | 97.8% | 2.17 |
| 32 x 32 | 44.9% | 89.8% | 94.9% | 66.1% | **66.7%** | 66.6% | 1.90 |

- 32 x 8: a memory link runs at 98% of its peak data rate. Bound by bandwidth with no wasted work,
  so the only way forward is moving less data (shared-memory tiling).
- 8 x 32: a memory unit is 98% busy while bandwidth is only 78% → the unit spends cycles on per-line
  bookkeeping (4 A lines + 1 B line per warp load) rather than moving bytes. Same hit rates and occupancy
  as 32 x 8, which rules those out. Confirms the "distinct lines per request" hypothesis.
- 32 x 32: occupancy hypothesis confirmed (66.7%, 1 block of 1024 threads per SM). Neither memory metric is
  saturated, so it's latency-bound: too few warps to hide memory latency. The lower L2 hit rate likely comes
  from fewer blocks running concurrently (less overlap between their A/B footprints); not verified.
- The earlier "bx x 16" sweep is superseded by this one (zsh word-splitting bug).

## cuBLAS reference (SGEMM, plain FP32 — CUBLAS_DEFAULT_MATH, no TF32)
| M x K x N | ms | GFLOP/s | errors |
|---|---|---|---|
| 1024 x 1024 x 1024 | 0.194  | 11,069.7 | 0 |
| 4096 x 4096 x 4096 | 11.737 | 11,709.7 | (skipped) |

- ~55-58% of the 3070's ~20.3 TFLOP/s spec peak (5888 FP32 lanes x 2 FLOP/FMA x ~1.73 GHz boost).
  Spec peak assumes every Ampere FP32 lane is busy; half of them share issue with INT32, and a GEMM
  inner loop also needs address and load instructions, so ~11-12 TFLOP/s is close to the practical ceiling.
- Gap at 1024^3: cuBLAS is 7.8x faster than our best kernel (swapped 32 x 8, 1423 GFLOP/s).

## Progression so far (1024^3)
| Kernel | GFLOP/s | % of cuBLAS |
|---|---|---|
| naive (row from x), 16 x 16 | 306 | 2.8% |
| naive, best shape 8 x 32    | 546 | 4.9% |
| swapped (col from x), 32 x 8 | 1,423 | 12.9% |
| tiled, shared memory, TILE 16 | 1,611 | 14.6% |
| tiled + coarsening (TILE 16, CF 4) | 1,691 | 15.3% |
| cuBLAS SGEMM | 11,070 | 100% |

## Tiled (shared memory, TILE_WIDTH 16)
| M x K x N | ms | GFLOP/s | errors |
|---|---|---|---|
| 1024 x 1024 x 1024 | 1.333 | 1,611.3 | 0 |
| 1026 x 1026 x 1026 | 1.387 | 1,557.3 | 0 (partial edge tiles, zero-padded) |
| 1017 x 632 x 261   | 0.229 | 1,466.3 | 0 after fixing the C bounds check |

- Only ~13% faster than swapped 32 x 8. Tiling cut global traffic ~16x, but swapped already had
  ~88% L1 hits, and the inner loop still does 2 shared-memory loads per FMA. On Ampere, shared memory
  and L1 are the same hardware, so the bottleneck moved to essentially the same unit.
- Shared-memory reads are conflict-free: sbA[threadRow][i] is a broadcast, sbB[i][threadCol] hits
  consecutive banks.
- Bugs hit on the way: row/col mixed between thread (x, y) and matrix [row][col]; `col < M` in the
  final bounds check. Square tests hid both; odd non-square shapes (1017 x 632 x 261) exposed them.
- Next: register blocking (each thread computes several outputs, more FMAs per shared load).

## Corner turning (B column-major, TILE 16)
| M x K x N | ms | GFLOP/s | errors |
|---|---|---|---|
| 1017 x 632 x 261 | 0.248 | 1,355.4 | 0 |

- ~8% slower than row-major tiled at the same shape (1,466). Likely cause: the transposed store
  sbB[threadCol][threadRow] has stride 16 across a warp -> only 2 banks per 16 threads, 8-way conflict.
  Fix to try: pad to sbB[TILE][TILE+1]. Fair baseline still to measure: column-major load without corner turning.

## Thread coarsening (TILE 16, COARSE_FACTOR horizontal tiles per block)
| CF | 1024^3 GFLOP/s | grid blocks (1024^3) | 1017 x 632 x 261 GFLOP/s | grid blocks (small) | wasted tiles (small) |
|---|---|---|---|---|---|
| 1 | 1,611.7 | 4096 | 1,454.6 | 1088 | 0 of 17 |
| 2 | 1,683.9 | 2048 | 1,399.6 | 576  | 1 of 18 |
| 4 | **1,690.8** | 1024 | 1,330.6 | 320  | 3 of 20 |
| 8 | 1,623.4 | 512  | 951.4   | 192  | 7 of 24 |

All runs: 0 errors. GPU holds 276 blocks at once (46 SMs x 6 blocks of 256 threads).
- 1024^3: sweet spot at CF 2-4 (+5%). Gain is small because each FMA still reads both sbA and sbB from
  shared memory; coarsening only saves A-tile loads, not inner-loop shared reads.
- Small shape: gets worse with every increase. Two causes stack: fewer blocks (CF 8 gives 192 < 276,
  so not even one full wave) and out-of-range tiles (261 cols = 17 tiles; CF 8 covers 24, ~29% wasted).
- Takeaway: coarsening trades parallelism for reuse. The reuse that matters next is in registers
  (each thread computing several outputs from one shared-memory read): register blocking.
