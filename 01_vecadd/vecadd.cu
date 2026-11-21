#include "common.cuh"
#include <cmath>
#include <vector>

__global__ void vecAdd(const float *__restrict__ a, const float *__restrict__ b,
                       float *__restrict__ c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = b[idx] + a[idx];
  }
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : (1 << 26);
  int threads = argc > 2 ? atoi(argv[2]) : 256;
  size_t bytes = (size_t)n * sizeof(float);

  printDevice();
  printf("n = %d (%.1f MB per array), threads/block = %d\n", n, bytes / 1e6,
         threads);

  std::vector<float> ha(n), hb(n), hc(n);

  for (int i = 0; i < n; i++) {
    ha[i] = sinf((float)i);
    hb[i] = cosf((float)i);
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, sizeof(float) * n));
  CUDA_CHECK(cudaMalloc(&db, sizeof(float) * n));
  CUDA_CHECK(cudaMalloc(&dc, sizeof(float) * n));
  CUDA_CHECK(
      cudaMemcpy(da, ha.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMemcpy(db, hb.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

  int blocks = (n + threads - 1) / threads;

  float ms = timeKernel([&] { vecAdd<<<blocks, threads>>>(da, db, dc, n); });

  CUDA_CHECK(
      cudaMemcpy(hc.data(), dc, sizeof(float) * n, cudaMemcpyDeviceToHost));
  int errors = 0;
  for (int i = 0; i < n; i++) {
    if (hc[i] != ha[i] + hb[i]) {
      if (errors < 5)
        fprintf(stderr, "mismatch at %d: got %f, expected %f\n", i, hc[i],
                ha[i] + hb[i]);
      errors++;
    }
  }

  double bw = gbps(3.0 * n * sizeof(float), ms);
  printf("Kernel time: %.4f ms | bandwidth: %.1f GB/s | errors: %d\n", ms, bw,
         errors);
  printf("CSV,%d,%d,%f,%f\n", n, threads, ms, bw);

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return errors == 0 ? 0 : 1;
}
