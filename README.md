# kernel_practice

Hand-written CUDA kernels, from vector add toward a fast GEMM. Each kernel gets a prediction,
a measurement, and an explanation.

## Layout
- `common/common.cuh` — shared plumbing: `CUDA_CHECK`, `idx`, `timeKernel`, `gbps`, `printDevice`
- `NN_<topic>/` — one folder per exercise; `*.cuh` for kernels, one `<topic>.cu` with `main`
- `scripts/` — `sweep.sh` (size sweep), `repeat.sh` (noise check)
- `results/` — CSVs and notes

## Build & run (on the GPU server)
    make                 # builds every */*.cu into bin/
    ./bin/transpose 8192 8192
    ncu --set full -o results/transpose ./bin/transpose 8192 8192

Programs print one machine-readable line starting with `CSV,` for the scripts.

## Develop here, run on the GPU server
`scripts/remote.sh` syncs this folder to the server with rsync, builds there, and streams output back.

    echo 'user@host' > .remote                          # once: your ssh target
    scripts/remote.sh run ./bin/vecadd                  # push + build + run
    scripts/remote.sh run scripts/sweep.sh bin/vecadd > results/vecadd_sweep.csv
    scripts/remote.sh run ncu -o results/vecadd ./bin/vecadd
    scripts/remote.sh pull                              # fetch files the server wrote to results/
    scripts/remote.sh ssh                               # interactive shell in the remote dir

The sync excludes `bin/`, `results/` and `.git/`, so server-side results are never overwritten.

## Roadmap
0. Vector add — bandwidth, L2 cliff (done: ~410 GB/s, 92% of 448 GB/s peak, L2 cliff at 4 MiB)
1. Transpose — naive (read- vs write-coalesced), copy baseline, shared-memory tiled, padded
2. Reductions — dot product, row softmax, online softmax (PMPP Ch 10)
3. Naive GEMM + cuBLAS baseline
4. Tiled GEMM (PMPP Ch 5)
5. Register blocking → warp tiling (Boehm worklog)
6. Tensor cores — WMMA, mma.sync, ldmatrix, cp.async double buffering
7. CUTLASS/CuTe or Triton; fused attention

## Hardware (RTX 3070)
448 GB/s DRAM peak · 4 MiB L2 · ~20 TFLOPS FP32 · ridge point ≈ 45 FLOP/byte

## Editor setup (clangd on macOS)
macOS has no CUDA toolkit, so clangd uses a header-only copy at ~/libs/cuda
(CUDA 12.9 runtime + nvcc crt + curand + CCCL headers, extracted from NVIDIA's pip wheels):

    tar xzf cuda-headers.tgz -C ~/libs     # creates ~/libs/cuda/{bin,include}

`.clangd` in this folder points clangd at that copy and treats `.cu`/`.cuh` as CUDA (sm_86).
