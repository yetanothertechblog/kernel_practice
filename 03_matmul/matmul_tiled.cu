#include "common.cuh"
#include <cmath>
#include <cstring>
#include <driver_types.h>
#include <vector>

#define TILE_WIDTH 32

__global__ void matmulTiled(const float *__restrict__ A,
                            const float *__restrict__ B, float *__restrict__ C,
                            int M, int K, int N) {

  __shared__ float sbA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sbB[TILE_WIDTH][TILE_WIDTH];

  // Thread (x,y) is still responsible for output (col, row), but we reduce
  // global memory accesses by collaboratively loading the block matrix of
  // TILE_WIDTHxTILE_WIDTH.
  // KEY CONFUSION HERE -> x and y are GRAPHICS coordinates.
  //
  //          x →  (columns)
  //       ┌──────────────
  //   y   │  (0,0) (1,0) (2,0)
  //   ↓   │  (0,1) (1,1) (2,1)
  // (rows)│
  //
  //
  // this is mega confusing when working with matrices where row is the first
  // co-ordinate is far more common Better to translate everything into Rows and
  // columns at the very top of the kernel so that there is little confusion!!

  int blockCol = blockIdx.x, blockRow = blockIdx.y, threadCol = threadIdx.x,
      threadRow = threadIdx.y;
  int col = blockCol * blockDim.x + threadCol;
  int row = blockRow * blockDim.y + threadRow;

  // run a loop for the block matrices of TILE_WIDTH x TILE_WIDTH, where we
  // first collaboratively load them
  float val = 0.0f;
  for (int tileIdx = 0; tileIdx < (K + TILE_WIDTH - 1) / TILE_WIDTH;
       tileIdx++) {

    int Arow = (blockRow * TILE_WIDTH + threadRow);
    int Acol = (tileIdx * TILE_WIDTH + threadCol);

    if (Arow < M && Acol < K) {
      sbA[threadRow][threadCol] = A[idx(Arow, Acol, K)];
    } else {
      sbA[threadRow][threadCol] = 0.0f;
    }
    int Brow = tileIdx * TILE_WIDTH + threadRow,
        Bcol = blockCol * TILE_WIDTH + threadCol;

    if (Brow < K && Bcol < N) {
      sbB[threadRow][threadCol] = B[idx(Brow, Bcol, N)];
    } else {
      sbB[threadRow][threadCol] = 0.0f;
    }

    __syncthreads();

    // now the entire matrix is collaboratively loaded. Take the local dot
    // product

    for (int innerTileIdx = 0; innerTileIdx < TILE_WIDTH; innerTileIdx++) {
      val += sbA[threadRow][innerTileIdx] * sbB[innerTileIdx][threadCol];
    }

    __syncthreads();
  }
  if (col < N && row < M) {
    C[idx(row, col, N)] = val;
  }
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

  printDevice();
  printf("C(%d x %d) = A(%d x %d) * B(%d x %d), block = (%d, %d)\n", M, N, M, K,
         K, N, TILE_WIDTH, TILE_WIDTH);

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

  dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
            (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  float ms =
      timeKernel([&] { matmulTiled<<<grid, block>>>(dA, dB, dC, M, K, N); });

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
  printf("CSV,%d,%d,%d,%d,%d,%f,%f\n", M, K, N, TILE_WIDTH, TILE_WIDTH, ms, gf);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return errors == 0 ? 0 : 1;
}
