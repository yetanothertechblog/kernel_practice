#include "common.cuh"
#include <cmath>
#include <cstring>
#include <vector>

constexpr int BM = 128; // rows of C per block
constexpr int BN = 128; // cols of C per block
constexpr int BK = 8;   // K-depth of each shared tile
constexpr int TM = 8;   // rows of each thread's result patch
constexpr int TN = 8;   // cols of each thread's result patch
constexpr int NUM_THREADS = BM * BN / (TM * TN);

// A: M x K (ld K)   B: K x N (ld N)   C: M x N (ld N), all row-major.
// 1D block of NUM_THREADS threads; block (bx, by) computes C tile
// [by*BM .. by*BM+BM) x [bx*BN .. bx*BN+BN).
// Shared tiles: As is BM x BK, Bs is BK x BN. Each thread computes a TM x TN
// patch and loads several elements of each shared tile.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmulBlocktile2D(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int K, int N) {

  float threadResults[TM * TN] = {0.0f};
  float Acache[TM] = {0.0f};
  float Bcache[TN] = {0.0f};

  // Loading BM * BK tile from A, B
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  int gridCol = blockIdx.x;
  int gridRow = blockIdx.y;

  A += K * BM *
       gridRow; // points to the top left corner of the A matrix we want to load
  B += BN * gridCol;

  C += N * (gridRow * BM) + gridCol * BN; // basically standard index only
                                          //
  int threadRowC = TM * (threadIdx.x / (BN / TN)); // top left corner of C
  int threadColC = TN * (threadIdx.x % (BN / TN));
  for (uint blockCol = 0; blockCol < K; blockCol += BK) {
    // load the As shared matrix
    for (uint idxOffset = threadIdx.x; idxOffset < BM * BK;
         idxOffset += NUM_THREADS) {
      As[idxOffset] = A[(idxOffset / BK) * K + (idxOffset % BK)];
    }
    for (uint idxOffset = threadIdx.x; idxOffset < BK * BN;
         idxOffset += NUM_THREADS) {
      Bs[idxOffset] = B[(idxOffset / BN) * N + (idxOffset % BN)];
    }
    __syncthreads();

    // advance A, B submartix pointers
    A += BK;
    B += N * BK;

    // a thread is responsible for a grid of TM * TN outputs in its column

    for (uint k = 0; k < BK; k++) {
      for (uint rowIdx = 0; rowIdx < TM; rowIdx++) {
        Acache[rowIdx] = As[(threadRowC + rowIdx) * BK + k];
      }
      for (uint colIdx = 0; colIdx < TN; colIdx++) {
        Bcache[colIdx] = Bs[(k)*BN + (threadColC + colIdx)];
      }

      for (uint rowIdx = 0; rowIdx < TM; rowIdx++) {
        for (uint colIdx = 0; colIdx < TN; colIdx++) {
          threadResults[rowIdx * TN + colIdx] +=
              Acache[rowIdx] * Bcache[colIdx];
        }
      }
    }
    __syncthreads();
  }

  for (uint resultIdx = 0; resultIdx < TM * TN; resultIdx++) {
    C[(threadRowC + resultIdx / TN) * N + (threadColC + resultIdx % TN)] =
        threadResults[resultIdx];
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
  static_assert((BM * BK) % NUM_THREADS == 0 && (BK * BN) % NUM_THREADS == 0,
                "each thread loads a whole number of elements per shared tile");
  printf("C(%d x %d) = A(%d x %d) * B(%d x %d), BM=%d BN=%d BK=%d TM=%d TN=%d, "
         "%d threads/block\n",
         M, N, M, K, K, N, BM, BN, BK, TM, TN, NUM_THREADS);

  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    fprintf(stderr,
            "This kernel has no bounds checks: M and N must be multiples of "
            "%d/%d, K a multiple of %d.\n",
            BM, BN, BK);
    return 1;
  }

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

  dim3 block(NUM_THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  float ms = timeKernel([&] {
    matmulBlocktile2D<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
  });

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
  printf("CSV,%d,%d,%d,%d,%d,%f,%f\n", M, K, N, BM, TM, ms, gf);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return errors == 0 ? 0 : 1;
}
