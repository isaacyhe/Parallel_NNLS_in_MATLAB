/**
 * @file nnls_admm_fp64_cuda.cu
 * @brief ADMM for NNLS, CUDA (FP64)
 *
 * Solves  min ||C*x - d||^2  subject to  x >= 0  via ADMM, GPU-resident.
 *
 * Splitting: min ||C*x - d||^2 + I_+(z)  s.t.  x = z
 * Updates (scaled form):
 *     x <- (2*C'*C + rho*I)^{-1} * (2*C'*d + rho*(z - u))
 *     z <- max(0, x + u)
 *     u <- u + (x - z)
 *
 * GPU layout:
 *   - C, d uploaded once and kept on device
 *   - cuBLAS dsyrk forms H = C^T*C (upper triangle)
 *   - Custom kernels build M = 2*H + rho*I in place
 *   - cuSOLVER dpotrf factors M = R^T*R
 *   - Per iter: cuSOLVER dpotrs solves M*x = rhs (two trsv inside)
 *   - Custom admmStep kernel does z = max(0, x+u); u += x - z fused
 *
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
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

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t st = call; \
    if (st != CUSOLVER_STATUS_SUCCESS) { \
        mexPrintf("cuSOLVER Error at %s:%d: %d\n", __FILE__, __LINE__, st); \
        mexErrMsgIdAndTxt("cuSOLVER:error", "cuSOLVER operation failed"); \
    } \
} while (0)

#ifdef USE_UNIFIED_MEMORY
  #define GPU_ALLOC(ptr, size) CUDA_CHECK(cudaMallocManaged(ptr, size))
#else
  #define GPU_ALLOC(ptr, size) CUDA_CHECK(cudaMalloc(ptr, size))
#endif

// Scale entire matrix in place: M[i] *= 2
__global__ void scale2Kernel(double* M, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) M[i] *= 2.0;
}

// Add rho to diagonal entries (column-major n x n)
__global__ void diagAddKernel(double* M, double rho, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) M[(size_t)j * (size_t)n + (size_t)j] += rho;
}

// rhs[j] = 2*q[j] + rho*(z[j] - u[j])
__global__ void buildRhsKernel(const double* q, const double* z, const double* u,
                               double rho, double* rhs, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) rhs[j] = 2.0 * q[j] + rho * (z[j] - u[j]);
}

// Fused: z = max(0, x+u);  u += x - z
__global__ void admmStepKernel(const double* x, double* u, double* z, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double xj = x[j];
        double uj = u[j];
        double zn = xj + uj;
        if (zn < 0.0) zn = 0.0;
        u[j] = uj + xj - zn;
        z[j] = zn;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("nnls_admm_fp64_cuda:input", "Two inputs required: C, d");

    int m = (int)mxGetM(prhs[0]);
    int n = (int)mxGetN(prhs[0]);
    const double* C_in = mxGetPr(prhs[0]);
    const double* d_in = mxGetPr(prhs[1]);

    const double rho      = 10.0;
    const int    max_iter = 500;

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t solver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

    auto start = chrono::high_resolution_clock::now();

    // ---- Upload C, d ----
    double *d_C, *d_d;
    GPU_ALLOC(&d_C, (size_t)m * (size_t)n * sizeof(double));
    GPU_ALLOC(&d_d, (size_t)m * sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_C, C_in, (size_t)m * (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d_in, (size_t)m * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- Allocate H (n x n), q (n), and ADMM state ----
    double *d_H, *d_q;
    GPU_ALLOC(&d_H, (size_t)n * (size_t)n * sizeof(double));
    GPU_ALLOC(&d_q, (size_t)n * sizeof(double));

    double *d_x, *d_z, *d_u;
    GPU_ALLOC(&d_x, (size_t)n * sizeof(double));
    GPU_ALLOC(&d_z, (size_t)n * sizeof(double));
    GPU_ALLOC(&d_u, (size_t)n * sizeof(double));
    CUDA_CHECK(cudaMemset(d_z, 0, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_u, 0, (size_t)n * sizeof(double)));

    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // ---- Form H = C^T * C (upper triangle) via cuBLAS dsyrk ----
    const double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                             n, m, &one, d_C, m, &zero, d_H, n));

    // ---- Form q = C^T * d ----
    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_T, m, n,
                             &one, d_C, m, d_d, 1, &zero, d_q, 1));

    // C and d no longer needed (Gram is in d_H, q is in d_q)
    cudaFree(d_C); d_C = nullptr;
    cudaFree(d_d); d_d = nullptr;

    // ---- Build M = 2*H + rho*I  (only upper triangle is read by potrf) ----
    int blk = 256;
    size_t total = (size_t)n * (size_t)n;
    int grid_full = (int)((total + blk - 1) / blk);
    int grid_n    = (n + blk - 1) / blk;
    scale2Kernel<<<grid_full, blk>>>(d_H, total);
    diagAddKernel<<<grid_n, blk>>>(d_H, rho, n);

    // ---- Cholesky factor: cusolverDnDpotrf  (upper) ----
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        solver_handle, CUBLAS_FILL_MODE_UPPER, n, d_H, n, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(double)));
    CUSOLVER_CHECK(cusolverDnDpotrf(
        solver_handle, CUBLAS_FILL_MODE_UPPER, n, d_H, n, d_work, lwork, d_info));
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
        mexErrMsgIdAndTxt("nnls_admm_fp64_cuda:chol",
            "cusolverDnDpotrf failed (info=%d)", h_info);

    // ---- ADMM loop ----
    for (int iter = 0; iter < max_iter; ++iter) {
        // d_x <- 2*q + rho*(z - u)
        buildRhsKernel<<<grid_n, blk>>>(d_q, d_z, d_u, rho, d_x, n);
        // x <- M^{-1} x  via dpotrs (in place)
        CUSOLVER_CHECK(cusolverDnDpotrs(
            solver_handle, CUBLAS_FILL_MODE_UPPER, n, 1, d_H, n, d_x, n, d_info));
        // z = max(0, x+u);  u += x - z
        admmStepKernel<<<grid_n, blk>>>(d_x, d_u, d_z, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Download z ----
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* x_out = mxGetPr(plhs[0]);
    CUDA_CHECK(cudaMemcpy(x_out, d_z, (size_t)n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cudaFree(d_H); cudaFree(d_q);
    cudaFree(d_x); cudaFree(d_z); cudaFree(d_u);
    cudaFree(d_work); cudaFree(d_info);
    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);

    mexPrintf("NNLS ADMM (FP64, CUDA + cuSOLVER) - Time: %lld us\n",
              (long long)duration.count());
}
