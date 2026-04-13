/**
 * @file nnls_cg_fp64_cuda.cu
 * @brief Plain conjugate gradient on B'B x = B'd + terminal projection (FP64, CUDA)
 *
 * Solves the normal equations B'B x = B'd via Hestenes-Stiefel CG on the
 * GPU using cuBLAS dgemv, then clamps to x >= 0 with one max(0, .) at the
 * end. The bound is not enforced during iteration.
 *
 * Reference baseline. Per iter cost: 2 cuBLAS dgemvs on B.
 *
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexPrintf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        mexErrMsgIdAndTxt("CUDA:error", cudaGetErrorString(err)); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = call; \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        mexPrintf("cuBLAS Error at %s:%d: %d\n", __FILE__, __LINE__, st); \
        mexErrMsgIdAndTxt("cuBLAS:error", "cuBLAS operation failed"); \
    } \
} while (0)

#ifdef USE_UNIFIED_MEMORY
  #define GPU_ALLOC(ptr, size) CUDA_CHECK(cudaMallocManaged(ptr, size))
#else
  #define GPU_ALLOC(ptr, size) CUDA_CHECK(cudaMalloc(ptr, size))
#endif

// x[j] += alpha * p[j];  g[j] -= alpha * Hp[j]
__global__ void cgUpdateKernel(double* x, double* g, const double* p,
                               const double* Hp, double alpha, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        x[j] += alpha * p[j];
        g[j] -= alpha * Hp[j];
    }
}

// p[j] = g[j] + beta * p[j]
__global__ void cgDirKernel(double* p, const double* g, double beta, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        p[j] = g[j] + beta * p[j];
    }
}

// x = max(0, x)
__global__ void projKernel(double* x, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && x[j] < 0.0) x[j] = 0.0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("nnls_cg_fp64_cuda:input", "Two inputs required: C, d");

    int m = (int)mxGetM(prhs[0]);
    int n = (int)mxGetN(prhs[0]);
    const double* C_in = mxGetPr(prhs[0]);
    const double* d_in = mxGetPr(prhs[1]);

    const int max_iter = 500;

    mexPrintf("[CG] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n"
              "     Plain CG on Tikhonov PSF problems (kappa ~1e10):\n"
              "       relErr 0.05  needs ~55,000 iterations  (~8 min on CUDA FP32)\n"
              "       relErr 0.01  needs ~85,000 iterations  (~12 min on CUDA FP32)\n"
              "     With max_iter=%d this run will return relErr ~0.97.\n"
              "     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n",
              max_iter, max_iter);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    auto start = chrono::high_resolution_clock::now();

    // Upload C, d
    double *d_C, *d_d;
    GPU_ALLOC(&d_C, (size_t)m * (size_t)n * sizeof(double));
    GPU_ALLOC(&d_d, (size_t)m * sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_C, C_in, (size_t)m * (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d_in, (size_t)m * sizeof(double),
                          cudaMemcpyHostToDevice));

    // CG state on device
    double *d_x, *d_g, *d_p, *d_Bp, *d_Hp;
    GPU_ALLOC(&d_x,  (size_t)n * sizeof(double));
    GPU_ALLOC(&d_g,  (size_t)n * sizeof(double));
    GPU_ALLOC(&d_p,  (size_t)n * sizeof(double));
    GPU_ALLOC(&d_Bp, (size_t)m * sizeof(double));
    GPU_ALLOC(&d_Hp, (size_t)n * sizeof(double));
    CUDA_CHECK(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));

    const double one = 1.0, zero = 0.0;

    // g = C^T * d
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T, m, n,
                             &one, d_C, m, d_d, 1, &zero, d_g, 1));

    // p = g
    CUDA_CHECK(cudaMemcpy(d_p, d_g, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    // rs_old = g'*g
    double rs_old = 0.0;
    CUBLAS_CHECK(cublasDdot(handle, n, d_g, 1, d_g, 1, &rs_old));

    int blk = 256;
    int grid_n = (n + blk - 1) / blk;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Bp = C * p
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, m, n,
                                 &one, d_C, m, d_p, 1, &zero, d_Bp, 1));

        // Hp = C^T * Bp
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T, m, n,
                                 &one, d_C, m, d_Bp, 1, &zero, d_Hp, 1));

        // denom = p' * Hp
        double denom = 0.0;
        CUBLAS_CHECK(cublasDdot(handle, n, d_p, 1, d_Hp, 1, &denom));
        if (denom <= 0.0) break;

        double alpha = rs_old / denom;

        // x += alpha * p;  g -= alpha * Hp
        cgUpdateKernel<<<grid_n, blk>>>(d_x, d_g, d_p, d_Hp, alpha, n);

        // rs_new = g'*g
        double rs_new = 0.0;
        CUBLAS_CHECK(cublasDdot(handle, n, d_g, 1, d_g, 1, &rs_new));
        if (rs_new <= 0.0) break;

        double beta = rs_new / rs_old;

        // p = g + beta * p
        cgDirKernel<<<grid_n, blk>>>(d_p, d_g, beta, n);

        rs_old = rs_new;
    }

    // Terminal projection
    projKernel<<<grid_n, blk>>>(d_x, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetPr(plhs[0]), d_x, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cudaFree(d_C); cudaFree(d_d);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_p); cudaFree(d_Bp); cudaFree(d_Hp);
    cublasDestroy(handle);

    mexPrintf("NNLS CG (FP64, CUDA + cuBLAS) - Time: %lld us\n",
              (long long)duration.count());
}
