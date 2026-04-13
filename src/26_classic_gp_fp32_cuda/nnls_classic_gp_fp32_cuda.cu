/**
 * @file nnls_classic_gp_fp32_cuda.cu
 * @brief Plain projected gradient for NNLS (FP32, CUDA)
 *
 * Single-precision plain GP. See nnls_classic_gp_fp64_cuda.cu for the
 * algorithm and rationale.
 *
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

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

__global__ void gpStepKernel(float* x, const float* g, float inv_L, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float xn = x[j] - inv_L * g[j];
        if (xn < 0.0f) xn = 0.0f;
        x[j] = xn;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("nnls_classic_gp_fp32_cuda:input", "Two inputs required: C, d");

    int m = (int)mxGetM(prhs[0]);
    int n = (int)mxGetN(prhs[0]);
    const double* C_in = mxGetPr(prhs[0]);
    const double* d_in = mxGetPr(prhs[1]);

    const int max_iter = 500;

    mexPrintf("[GP] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n"
              "     Plain projected gradient on Tikhonov PSF problems (kappa ~1e10):\n"
              "       relErr 0.05  needs ~6.7e10 iterations  (>>years on any hardware)\n"
              "       relErr 0.01  needs ~1.0e11 iterations  (>>years on any hardware)\n"
              "     This algorithm class cannot reach low error on these problems.\n"
              "     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n",
              max_iter);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    auto start = chrono::high_resolution_clock::now();

    // Convert C, d to FP32
    vector<float> C32((size_t)m * (size_t)n);
    vector<float> d32((size_t)m);
    for (size_t i = 0; i < (size_t)m * (size_t)n; ++i) C32[i] = (float)C_in[i];
    for (int i = 0; i < m; ++i) d32[(size_t)i] = (float)d_in[i];

    float *d_C, *d_d;
    GPU_ALLOC(&d_C, (size_t)m * (size_t)n * sizeof(float));
    GPU_ALLOC(&d_d, (size_t)m * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_C, C32.data(), (size_t)m * (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d32.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice));
    C32 = vector<float>(); d32 = vector<float>();

    float *d_x, *d_g, *d_r, *d_v, *d_Cv, *d_Hv;
    GPU_ALLOC(&d_x,  (size_t)n * sizeof(float));
    GPU_ALLOC(&d_g,  (size_t)n * sizeof(float));
    GPU_ALLOC(&d_r,  (size_t)m * sizeof(float));
    GPU_ALLOC(&d_v,  (size_t)n * sizeof(float));
    GPU_ALLOC(&d_Cv, (size_t)m * sizeof(float));
    GPU_ALLOC(&d_Hv, (size_t)n * sizeof(float));
    CUDA_CHECK(cudaMemset(d_x, 0, (size_t)n * sizeof(float)));

    const float one = 1.0f, zero = 0.0f, neg_one = -1.0f;

    // Power iteration on C
    {
        vector<float> v_init((size_t)n);
        mt19937 rng(0);
        normal_distribution<float> dist(0.0f, 1.0f);
        float s2 = 0.0f;
        for (int j = 0; j < n; ++j) { v_init[(size_t)j] = dist(rng); s2 += v_init[(size_t)j]*v_init[(size_t)j]; }
        float s = sqrtf(s2);
        if (s > 0.0f) for (int j = 0; j < n; ++j) v_init[(size_t)j] /= s;
        CUDA_CHECK(cudaMemcpy(d_v, v_init.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    }
    float L = 0.0f;
    for (int k = 0; k < 30; ++k) {
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n, &one, d_C, m, d_v, 1, &zero, d_Cv, 1));
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n, &one, d_C, m, d_Cv, 1, &zero, d_Hv, 1));
        float L_new = 0.0f;
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_Hv, 1, &L_new));
        if (L_new <= 0.0f) { L_new = 1.0f; L = L_new; break; }
        float inv = 1.0f / L_new;
        CUDA_CHECK(cudaMemcpy(d_v, d_Hv, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasSscal(handle, n, &inv, d_v, 1));
        if (k > 0 && fabsf(L_new - L) < 1e-4f * L_new) { L = L_new; break; }
        L = L_new;
    }
    L *= 1.01f;
    const float inv_L = 1.0f / L;

    int blk = 256;
    int grid_n = (n + blk - 1) / blk;

    for (int iter = 0; iter < max_iter; ++iter) {
        CUDA_CHECK(cudaMemcpy(d_r, d_d, (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n, &one, d_C, m, d_x, 1, &neg_one, d_r, 1));
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n, &one, d_C, m, d_r, 1, &zero, d_g, 1));
        gpStepKernel<<<grid_n, blk>>>(d_x, d_g, inv_L, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* x_out = mxGetPr(plhs[0]);
    vector<float> x_host((size_t)n);
    CUDA_CHECK(cudaMemcpy(x_host.data(), d_x, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; ++j) x_out[j] = (double)x_host[(size_t)j];

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cudaFree(d_C); cudaFree(d_d);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_r);
    cudaFree(d_v); cudaFree(d_Cv); cudaFree(d_Hv);
    cublasDestroy(handle);

    mexPrintf("NNLS Classic GP (FP32, CUDA + cuBLAS) - Time: %lld us\n",
              (long long)duration.count());
}
