/**
 * @file nnls_admm_fp32_cuda.cu
 * @brief ADMM for NNLS, CUDA (FP32)
 *
 * Solves  min ||C*x - d||^2  subject to  x >= 0  via ADMM, GPU-resident.
 *
 * Single precision: cuBLAS ssyrk + cuSOLVER spotrf + spotrs.
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

__global__ void scale2Kernel(float* M, size_t total) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) M[i] *= 2.0f;
}

__global__ void diagAddKernel(float* M, float rho, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) M[(size_t)j * (size_t)n + (size_t)j] += rho;
}

__global__ void buildRhsKernel(const float* q, const float* z, const float* u,
                               float rho, float* rhs, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) rhs[j] = 2.0f * q[j] + rho * (z[j] - u[j]);
}

__global__ void admmStepKernel(const float* x, float* u, float* z, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float xj = x[j];
        float uj = u[j];
        float zn = xj + uj;
        if (zn < 0.0f) zn = 0.0f;
        u[j] = uj + xj - zn;
        z[j] = zn;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("nnls_admm_fp32_cuda:input", "Two inputs required: C, d");

    int m = (int)mxGetM(prhs[0]);
    int n = (int)mxGetN(prhs[0]);
    const double* C_in = mxGetPr(prhs[0]);
    const double* d_in = mxGetPr(prhs[1]);

    // FP32: cuSOLVER Spotrf is more sensitive to near-PD matrices than
    // host LAPACK on PSF-class problems. Bump rho from 10 to 15 to make
    // the Cholesky factor numerically clean. Convergence floor is ~7e-2
    // due to accumulated FP32 round-off in the iterative trsv path —
    // additional iterations beyond ~500 don't improve the result.
    const float rho      = 15.0f;
    const int   max_iter = 500;

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t solver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    // Force strict IEEE FP32 (disable TF32 tensor ops on Ampere+).
    // TF32's 10-bit mantissa wrecks ssyrk accumulation precision and the
    // resulting Cholesky factor on PSF-class problems.
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH));
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

    auto start = chrono::high_resolution_clock::now();

    // ---- Convert C, d to FP32 on host ----
    vector<float> C32((size_t)m * (size_t)n);
    vector<float> d32((size_t)m);
    for (size_t i = 0; i < (size_t)m * (size_t)n; ++i) C32[i] = (float)C_in[i];
    for (int i = 0; i < m; ++i) d32[(size_t)i] = (float)d_in[i];

    // ---- Upload to device ----
    float *d_C, *d_d;
    GPU_ALLOC(&d_C, (size_t)m * (size_t)n * sizeof(float));
    GPU_ALLOC(&d_d, (size_t)m * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_C, C32.data(), (size_t)m * (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d32.data(), (size_t)m * sizeof(float),
                          cudaMemcpyHostToDevice));
    C32 = vector<float>(); d32 = vector<float>();

    // ---- Allocate H, q, ADMM state ----
    float *d_H, *d_q;
    GPU_ALLOC(&d_H, (size_t)n * (size_t)n * sizeof(float));
    GPU_ALLOC(&d_q, (size_t)n * sizeof(float));

    float *d_x, *d_z, *d_u;
    GPU_ALLOC(&d_x, (size_t)n * sizeof(float));
    GPU_ALLOC(&d_z, (size_t)n * sizeof(float));
    GPU_ALLOC(&d_u, (size_t)n * sizeof(float));
    CUDA_CHECK(cudaMemset(d_z, 0, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_u, 0, (size_t)n * sizeof(float)));

    int *d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Zero d_H first so the lower triangle is well-defined (cudaMalloc
    // returns uninitialized memory). Cholesky only reads upper, but
    // scale2Kernel touches the full matrix and we want to avoid any
    // NaN/Inf garbage propagating.
    CUDA_CHECK(cudaMemset(d_H, 0, (size_t)n * (size_t)n * sizeof(float)));

    // ---- Form H = C^T * C (upper triangle) via cuBLAS ssyrk ----
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSsyrk(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                             n, m, &one, d_C, m, &zero, d_H, n));

    // ---- Form q = C^T * d ----
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, m, n,
                             &one, d_C, m, d_d, 1, &zero, d_q, 1));

    // C, d no longer needed
    cudaFree(d_C); d_C = nullptr;
    cudaFree(d_d); d_d = nullptr;

    // ---- Build M = 2*H + rho*I ----
    int blk = 256;
    size_t total = (size_t)n * (size_t)n;
    int grid_full = (int)((total + blk - 1) / blk);
    int grid_n    = (n + blk - 1) / blk;
    scale2Kernel<<<grid_full, blk>>>(d_H, total);
    diagAddKernel<<<grid_n, blk>>>(d_H, rho, n);

    // ---- Cholesky factor: cusolverDnSpotrf ----
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
        solver_handle, CUBLAS_FILL_MODE_UPPER, n, d_H, n, &lwork));
    float* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSpotrf(
        solver_handle, CUBLAS_FILL_MODE_UPPER, n, d_H, n, d_work, lwork, d_info));
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
        mexErrMsgIdAndTxt("nnls_admm_fp32_cuda:chol",
            "cusolverDnSpotrf failed (info=%d)", h_info);

    // ---- ADMM loop ----
    for (int iter = 0; iter < max_iter; ++iter) {
        buildRhsKernel<<<grid_n, blk>>>(d_q, d_z, d_u, rho, d_x, n);
        CUSOLVER_CHECK(cusolverDnSpotrs(
            solver_handle, CUBLAS_FILL_MODE_UPPER, n, 1, d_H, n, d_x, n, d_info));
        admmStepKernel<<<grid_n, blk>>>(d_x, d_u, d_z, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Download z and convert to double ----
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* x_out = mxGetPr(plhs[0]);
    vector<float> z_host((size_t)n);
    CUDA_CHECK(cudaMemcpy(z_host.data(), d_z, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j) x_out[j] = (double)z_host[(size_t)j];

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cudaFree(d_H); cudaFree(d_q);
    cudaFree(d_x); cudaFree(d_z); cudaFree(d_u);
    cudaFree(d_work); cudaFree(d_info);
    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);

    mexPrintf("NNLS ADMM (FP32, CUDA + cuSOLVER) - Time: %lld us\n",
              (long long)duration.count());
}
