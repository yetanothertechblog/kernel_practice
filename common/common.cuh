#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA ERR: %s at %s:%d\n", cudaGetErrorString(err_),     \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Row-major index: ld = leading dimension = row length (number of columns).
__host__ __device__ inline int idx(int r, int c, int ld) { return r * ld + c; }

template <typename Launch> float timeKernel(Launch launch, int iters = 20) {
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

inline double gbps(double bytesMoved, float ms) {
  return bytesMoved / (ms * 1e-3) / 1e9;
}

// Throughput in GFLOP/s. A GEMM of M x K times K x N does 2*M*N*K FLOPs.
inline double gflops(double flops, float ms) {
  return flops / (ms * 1e-3) / 1e9;
}

// Prints the GPU name, SM count, L2 size and theoretical DRAM bandwidth.
inline void printDevice(int dev = 0) {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

  int memClockKHz = 0, busWidthBits = 0;

  CUDA_CHECK(
      cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&busWidthBits,
                                    cudaDevAttrGlobalMemoryBusWidth, dev));
  double peakGBps = 2.0 * memClockKHz * 1e3 * (busWidthBits / 8.0) / 1e9;

  printf("Device: %s (sm_%d%d, %d SMs, L2 %.1f MiB, %.1f GiB, peak DRAM ~%.0f "
         "GB/s)\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.l2CacheSize / (1024.0 * 1024.0),
         p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), peakGBps);
}
