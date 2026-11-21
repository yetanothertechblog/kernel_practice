#include "common.cuh"
#include <cmath>
#include <vector>

__global__ void transposeMat(const float *__restrict__ a, float *__restrict__ b,
                             int m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    b[idx(j, i, m)] = a[idx(i, j, n)];
  }
}

int main(int argc, char **argv) {
  int m = argc > 1 ? atoi(argv[1]) : 8192;
  int n = argc > 2 ? atoi(argv[2]) : 8192;

  size_t bytes = (size_t)m * (size_t)n * sizeof(float);

  printDevice();
  printf("A(%d x %d) -> B(%d x %d), %.1f MB per matrix\n", m, n, n, m,
         bytes / 1e6);

  std::vector<float> ha((size_t)m * n), hb((size_t)m * n);
  for (size_t i = 0; i < ha.size(); i++) {
    ha[i] = sinf((float)i);
  }

  float *da, *db;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(32, 8);
  dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

  float ms = timeKernel([&] { transposeMat<<<grid, block>>>(da, db, m, n); });

  CUDA_CHECK(cudaMemcpy(hb.data(), db, bytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (hb[idx(j, i, m)] != ha[idx(i, j, n)]) {
        errors++;
      }
    }
  }

  double bw = gbps(2.0 * bytes, ms);
  printf("Kernel time: %.4f ms | bandwidth: %.1f GB/s | errors: %d\n", ms, bw,
         errors);
  printf("CSV,%d,%d,%f,%f\n", m, n, ms, bw);

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  return errors == 0 ? 0 : 1;
}
