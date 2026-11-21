#include "common.cuh"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <vector>

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st_ = (call);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS ERR: %d at %s:%d\n", (int)st_, __FILE__,         \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Reference SGEMM. cuBLAS is column-major; our matrices are row-major.
// A row-major M x N matrix has the same bytes as a column-major N x M matrix
// (its transpose), so C = A * B row-major is computed as C^T = B^T * A^T
// column-major: pass B first, A second, and swap M and N. No data is moved.
void matmulCublas(cublasHandle_t h, const float *A, const float *B, float *C,
                  int M, int K, int N) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                           A, K, &beta, C, N));
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
  printf("C(%d x %d) = A(%d x %d) * B(%d x %d), cuBLAS SGEMM (FP32, no TF32)\n",
         M, N, M, K, K, N);

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

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // Plain FP32 on CUDA cores, so the comparison with our kernels is fair.
  // CUBLAS_TF32_TENSOR_OP_MATH would use tensor cores at reduced precision.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  float ms = timeKernel([&] { matmulCublas(handle, dA, dB, dC, M, K, N); });

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
  printf("CSV,%d,%d,%d,%d,%d,%f,%f\n", M, K, N, 0, 0, ms, gf);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return errors == 0 ? 0 : 1;
}
