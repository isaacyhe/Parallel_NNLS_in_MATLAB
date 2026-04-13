/**
 * @file nnls_cg_fp32_cuda.cu
 * @brief Plain conjugate gradient on B'B x = B'd + terminal projection (FP32, CUDA)
 *
 * Single-precision plain CG. See nnls_cg_fp64_cuda.cu for the algorithm.
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

__global__ void cgUpdateKernel(float* x, float* g, const float* p,
                               const float* Hp, float alpha, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        x[j] += alpha * p[j];
        g[j] -= alpha * Hp[j];
    }
}

__global__ void cgDirKernel(float* p, const float* g, float beta, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        p[j] = g[j] + beta * p[j];
    }
}

__global__ void projKernel(float* x, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && x[j] < 0.0f) x[j] = 0.0f;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("nnls_cg_fp32_cuda:input", "Two inputs required: C, d");

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

    // Convert C, d to FP32 on host
    vector<float> C32((size_t)m * (size_t)n);
    vector<float> d32((size_t)m);
    for (size_t i = 0; i < (size_t)m * (size_t)n; ++i) C32[i] = (float)C_in[i];
    for (int i = 0; i < m; ++i) d32[(size_t)i] = (float)d_in[i];

    // Upload
    float *d_C, *d_d;
    GPU_ALLOC(&d_C, (size_t)m * (size_t)n * sizeof(float));
    GPU_ALLOC(&d_d, (size_t)m * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_C, C32.data(), (size_t)m * (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d32.data(), (size_t)m * sizeof(float),
                          cudaMemcpyHostToDevice));
    C32 = vector<float>(); d32 = vector<float>();

    // CG state
    float *d_x, *d_g, *d_p, *d_Bp, *d_Hp;
    GPU_ALLOC(&d_x,  (size_t)n * sizeof(float));
    GPU_ALLOC(&d_g,  (size_t)n * sizeof(float));
    GPU_ALLOC(&d_p,  (size_t)n * sizeof(float));
    GPU_ALLOC(&d_Bp, (size_t)m * sizeof(float));
    GPU_ALLOC(&d_Hp, (size_t)n * sizeof(float));
    CUDA_CHECK(cudaMemset(d_x, 0, (size_t)n * sizeof(float)));

    const float one = 1.0f, zero = 0.0f;

    // g = C^T * d
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n,
                             &one, d_C, m, d_d, 1, &zero, d_g, 1));
    CUDA_CHECK(cudaMemcpy(d_p, d_g, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    float rs_old = 0.0f;
    CUBLAS_CHECK(cublasSdot(handle, n, d_g, 1, d_g, 1, &rs_old));

    int blk = 256;
    int grid_n = (n + blk - 1) / blk;

    for (int iter = 0; iter < max_iter; ++iter) {
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n,
                                 &one, d_C, m, d_p, 1, &zero, d_Bp, 1));

        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n,
                                 &one, d_C, m, d_Bp, 1, &zero, d_Hp, 1));

        float denom = 0.0f;
        CUBLAS_CHECK(cublasSdot(handle, n, d_p, 1, d_Hp, 1, &denom));
        if (denom <= 0.0f) break;

        float alpha = rs_old / denom;
        cgUpdateKernel<<<grid_n, blk>>>(d_x, d_g, d_p, d_Hp, alpha, n);

        float rs_new = 0.0f;
        CUBLAS_CHECK(cublasSdot(handle, n, d_g, 1, d_g, 1, &rs_new));
        if (rs_new <= 0.0f) break;

        float beta = rs_new / rs_old;
        cgDirKernel<<<grid_n, blk>>>(d_p, d_g, beta, n);

        rs_old = rs_new;
    }

    projKernel<<<grid_n, blk>>>(d_x, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download as double
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* x_out = mxGetPr(plhs[0]);
    vector<float> x_host((size_t)n);
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x, (size_t)n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j) x_out[j] = (double)x_host[(size_t)j];

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cudaFree(d_C); cudaFree(d_d);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_p); cudaFree(d_Bp); cudaFree(d_Hp);
    cublasDestroy(handle);

    mexPrintf("NNLS CG (FP32, CUDA + cuBLAS) - Time: %lld us\n",
              (long long)duration.count());
}
