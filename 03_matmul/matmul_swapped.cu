#include "common.cuh"
#include <cmath>
#include <cstring>
#include <driver_types.h>
#include <vector>

__global__ void matmulNaive(const float *__restrict__ A,
                            const float *__restrict__ B, float *__restrict__ C,
                            int M, int K, int N) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  // in a block the read patterns are A[row*K + k] and B[k*N + col]
  // as col comes from the threadIdx.x, B reads are coalesced. whereas A reads
  // are BROADCAST Both are very efficient
  if (row < M && col < N) {
    float val = 0;
    for (int k = 0; k < K; k++) {
      val += A[idx(row, k, K)] * B[idx(k, col, N)];
    }
    C[idx(row, col, N)] = val;
  }
  // the writes are C[row * N + col] warps are 32 size and built from
  // consecutive thread IDs (x varies the fastest)
  // this is a coalesced write too!
  // also this is 4x faster
}

void matmulCPU(const std::vector<float> &A, const std::vector<float> &B,
               std::vector<double> &C, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[idx(i, j, N)] += (double)A[idx(i, k, K)] * B[idx(k, j, N)];
      }
    }
  }
}

int main(int argc, char **argv) {
  bool verify = true;
  if (argc > 1 && strcmp(argv[argc - 1], "--no-verify") == 0) {
    verify = false;
    argc--;
  }

  int M = argc > 1 ? atoi(argv[1]) : 1024;
  int K = argc > 2 ? atoi(argv[2]) : 1024;
  int N = argc > 3 ? atoi(argv[3]) : 1024;
  int bx = argc > 4 ? atoi(argv[4]) : 16;
  int by = argc > 5 ? atoi(argv[5]) : 16;

  printDevice();
  printf("C(%d x %d) = A(%d x %d) * B(%d x %d), block = (%d, %d)\n", M, N, M, K,
         K, N, bx, by);

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  for (size_t i = 0; i < hA.size(); i++)
    hA[i] = sinf((float)i);
  for (size_t i = 0; i < hB.size(); i++)
    hB[i] = cosf((float)i);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)M * K * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(dB, hB.data(), (size_t)K * N * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 block(bx, by, 1);
  dim3 grid((N + bx - 1) / bx, (M + by - 1) / by, 1);

  float ms =
      timeKernel([&] { matmulNaive<<<grid, block>>>(dA, dB, dC, M, K, N); });

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int errors = 0;

  if (verify) {
    std::vector<double> ref((size_t)M * N);
    matmulCPU(hA, hB, ref, M, K, N);

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        double err = std::abs(hC[idx(i, j, N)] - ref[idx(i, j, N)]);
        if (err > 1e-3 * (1 + std::abs(ref[idx(i, j, N)]))) {
          errors++;
        }
      }
    }
  } else {
    printf("(verification skipped)\n");
  }

  double gf = gflops(2.0 * M * N * K, ms);
  printf("Kernel time: %.4f ms | %.1f GFLOP/s | errors: %d\n", ms, gf, errors);
  printf("CSV,%d,%d,%d,%d,%d,%f,%f\n", M, K, N, bx, by, ms, gf);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return errors == 0 ? 0 : 1;
}
