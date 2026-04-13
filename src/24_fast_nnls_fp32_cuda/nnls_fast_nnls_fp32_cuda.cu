/**
 * @file nnls_fast_nnls_fp32_cuda.cu
 * @brief FAST-NNLS (Cobb et al., IEEE BigData 2025) with CUDA (FP32)
 *
 * GPU-resident implementation: C and d stay on GPU throughout. cuBLAS for
 * pre-computation (Zty) and gradient updates (two-gemv to avoid condition
 * number squaring). cuSOLVER for QR factorization of sub-problems.
 *
 * References:
 *   - Cobb et al. (2025). FAST-NNLS: A fast and exact non-negative least
 *     squares algorithm. IEEE BigData.
 *
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>
#include <limits>
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

#ifdef USE_UM_PREFETCH
  static inline void gpuPrefetch(void* ptr, size_t size, cudaStream_t stream = 0) {
      int dev; cudaGetDevice(&dev);
      cudaMemPrefetchAsync(ptr, size, dev, stream);
  }
  #define GPU_PREFETCH(ptr, size) gpuPrefetch(ptr, size)
#else
  #define GPU_PREFETCH(ptr, size) ((void)0)
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2)
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp32_cuda:input",
            "Two inputs required: C, d");

    int m = (int)mxGetM(prhs[0]);
    int n = (int)mxGetN(prhs[0]);
    const double* C_dbl = mxGetPr(prhs[0]);
    const double* d_dbl = mxGetPr(prhs[1]);

    // Convert MATLAB double to float (column-major preserved)
    vector<float> C_f((size_t)m * n);
    vector<float> d_f(m);
    for (size_t i = 0; i < (size_t)m * n; ++i)
        C_f[i] = (float)C_dbl[i];
    for (int i = 0; i < m; ++i)
        d_f[i] = (float)d_dbl[i];

    float tol = 1e-4f;
    int max_iter = 30 * n;
    float theta_add = 0.5f;
    float theta_rem = 0.5f;

    // --- Create handles ---
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t solver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

    // --- Allocate persistent GPU buffers ---
    float *d_C, *d_d, *d_Zty;
    float *d_x, *d_r, *d_w;
    float *d_C_sub, *d_d_work, *d_tau, *d_work;
    int *d_info;

    GPU_ALLOC(&d_C,     (size_t)m * n * sizeof(float));
    GPU_ALLOC(&d_d,     (size_t)m * sizeof(float));
    GPU_ALLOC(&d_Zty,   (size_t)n * sizeof(float));
    GPU_ALLOC(&d_x,     (size_t)n * sizeof(float));
    GPU_ALLOC(&d_r,     (size_t)m * sizeof(float));
    GPU_ALLOC(&d_w,     (size_t)n * sizeof(float));
    GPU_ALLOC(&d_C_sub, (size_t)m * n * sizeof(float));
    GPU_ALLOC(&d_d_work,(size_t)m * sizeof(float));
    GPU_ALLOC(&d_tau,   (size_t)n * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // --- Upload data ---
    CUDA_CHECK(cudaMemcpy(d_C, C_f.data(), (size_t)m * n * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PREFETCH(d_C, (size_t)m * n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_d, d_f.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PREFETCH(d_d, (size_t)m * sizeof(float));

    // --- Pre-compute Zty = C'*d on GPU (skip ZtZ for FP32 — condition number squaring) ---
    float alpha_blas = 1.0f, beta_blas = 0.0f;
    CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, m, n,
                             &alpha_blas, d_C, m, d_d, 1, &beta_blas, d_Zty, 1));

    // Download Zty for initial w
    vector<float> Zty(n);
    CUDA_CHECK(cudaMemcpy(Zty.data(), d_Zty, n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Query cuSOLVER workspace for max sub-problem size (m x n) ---
    int lwork_geqrf = 0, lwork_ormqr = 0;
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver_handle, m, n, d_C_sub, m, &lwork_geqrf));
    CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                                m, 1, n, d_C_sub, m, d_tau,
                                                d_d_work, m, &lwork_ormqr));
    int lwork = max(lwork_geqrf, lwork_ormqr);
    GPU_ALLOC(&d_work, (size_t)lwork * sizeof(float));

    auto start = chrono::high_resolution_clock::now();

    // === FAST-NNLS main algorithm ===
    vector<float> x(n, 0.0f);
    vector<bool> P(n, false);
    vector<float> w(Zty);
    vector<float> s(n, 0.0f);
    vector<float> h_R((size_t)m * n);   // host buffer for QR R factor
    vector<float> h_rhs(m);             // host buffer for Q^T * d

    int iter = 0;

    while (true) {
        float max_w = -numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            if (!P[j] && w[j] > max_w) max_w = w[j];
        }
        if (max_w <= tol) break;

        // BATCH ADD
        float t_add = max_w * theta_add;
        for (int j = 0; j < n; ++j) {
            if (!P[j] && w[j] > t_add)
                P[j] = true;
        }

        while (true) {
            iter++;
            if (iter > max_iter) break;

            vector<int> idx_P;
            for (int j = 0; j < n; ++j)
                if (P[j]) idx_P.push_back(j);
            int np = (int)idx_P.size();
            if (np == 0) break;

            // Extract active columns via D2D copies
            for (int i = 0; i < np; ++i) {
                int j = idx_P[i];
                CUDA_CHECK(cudaMemcpy(d_C_sub + (size_t)i * m, d_C + (size_t)j * m,
                                     m * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            // Fresh copy of d
            CUDA_CHECK(cudaMemcpy(d_d_work, d_d, (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice));

            // QR factorize
            CUSOLVER_CHECK(cusolverDnSgeqrf(solver_handle, m, np, d_C_sub, m,
                                            d_tau, d_work, lwork, d_info));

            // Apply Q^T to d_d_work
            CUSOLVER_CHECK(cusolverDnSormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                            m, 1, np, d_C_sub, m, d_tau,
                                            d_d_work, m, d_work, lwork, d_info));

            // Download R factor and Q^T*d for safe back-substitution on host
            CUDA_CHECK(cudaMemcpy(h_R.data(), d_C_sub, (size_t)m * np * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_rhs.data(), d_d_work, np * sizeof(float),
                                 cudaMemcpyDeviceToHost));

            // Back-substitution with near-zero diagonal safeguard
            vector<float> s_sub(np);
            for (int i = np - 1; i >= 0; --i) {
                float sum = h_rhs[i];
                for (int j = i + 1; j < np; ++j)
                    sum -= h_R[i + (size_t)j * m] * s_sub[j];
                s_sub[i] = (fabsf(h_R[i + (size_t)i * m]) > 1e-7f)
                    ? sum / h_R[i + (size_t)i * m] : 0.0f;
            }

            for (int j = 0; j < n; ++j) s[j] = 0.0f;
            for (int i = 0; i < np; ++i) s[idx_P[i]] = s_sub[i];

            bool has_infeasible = false;
            for (int i = 0; i < np; ++i) {
                if (s[idx_P[i]] <= 0.0f) { has_infeasible = true; break; }
            }
            if (!has_infeasible) break;

            // BATCH REMOVE
            float min_s = numeric_limits<float>::infinity();
            for (int i = 0; i < np; ++i) {
                if (s[idx_P[i]] < min_s) min_s = s[idx_P[i]];
            }
            float t_rem_val = min_s * theta_rem;

            float alpha = numeric_limits<float>::infinity();
            for (int i = 0; i < np; ++i) {
                int j = idx_P[i];
                if (s[j] < t_rem_val) {
                    float a = x[j] / (x[j] - s[j]);
                    if (a < alpha) alpha = a;
                }
            }

            for (int j = 0; j < n; ++j)
                x[j] += alpha * (s[j] - x[j]);

            for (int j = 0; j < n; ++j) {
                if (P[j] && fabsf(x[j]) < tol && s[j] <= 0.0f) {
                    P[j] = false;
                    x[j] = 0.0f;
                }
            }
        }

        for (int j = 0; j < n; ++j) x[j] = s[j];

        // Gradient: w = C'*(d - C*x) via two cuBLAS gemv calls (avoids cond# squaring)
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        // r = d (D2D copy)
        CUDA_CHECK(cudaMemcpy(d_r, d_d, (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice));

        // r = d - C*x
        float neg_one = -1.0f, one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, m, n,
                                 &neg_one, d_C, m, d_x, 1, &one, d_r, 1));

        // w = C^T * r
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, m, n,
                                 &one, d_C, m, d_r, 1, &zero, d_w, 1));

        CUDA_CHECK(cudaMemcpy(w.data(), d_w, n * sizeof(float), cudaMemcpyDeviceToHost));

        if (iter > max_iter) break;
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    for (int i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;

    // --- Cleanup GPU ---
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_Zty));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_C_sub));
    CUDA_CHECK(cudaFree(d_d_work));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    // --- Output (float -> double for MATLAB) ---
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (int i = 0; i < n; ++i) out[i] = (double)x[i];

    mexPrintf("NNLS FAST-NNLS (FP32, CUDA) - Execution Time: %lld microseconds\n", duration.count());
}
