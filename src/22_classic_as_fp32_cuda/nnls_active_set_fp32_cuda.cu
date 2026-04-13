/**
 * @file nnls_active_set_fp32_cuda.cu
 * @brief Non-Negative Least Squares solver using Active Set method with CUDA (FP32)
 *
 * GPU-resident implementation: data stays on GPU, cuBLAS for matrix-vector ops,
 * cuSOLVER for QR factorization of sub-problems. No per-iteration alloc/free.
 *
 * The solver finds x that minimizes ||Cx - d||^2 subject to x >= 0
 *
 * References:
 *   - Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data Parallelism.
 *
 * @date 2025
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexPrintf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(err)); \
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
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsActiveSetCUDA:input",
                         "Two input arguments required: matrix C and vector d");
    }

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

    float tol = 1e-6f;
    int max_outer_iter = 3 * n;
    int max_inner_iter = 3 * n;

    // --- Create handles ---
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t solver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

    // --- Allocate persistent GPU buffers ---
    float *d_C, *d_d, *d_x, *d_r, *d_g;
    float *d_C_sub, *d_d_work, *d_tau, *d_work;
    int *d_info;

    GPU_ALLOC(&d_C,     (size_t)m * n * sizeof(float));
    GPU_ALLOC(&d_d,     (size_t)m * sizeof(float));
    GPU_ALLOC(&d_x,     (size_t)n * sizeof(float));
    GPU_ALLOC(&d_r,     (size_t)m * sizeof(float));
    GPU_ALLOC(&d_g,     (size_t)n * sizeof(float));
    GPU_ALLOC(&d_C_sub, (size_t)m * n * sizeof(float));
    GPU_ALLOC(&d_d_work,(size_t)m * sizeof(float));
    GPU_ALLOC(&d_tau,   (size_t)n * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // --- Upload data once ---
    CUDA_CHECK(cudaMemcpy(d_C, C_f.data(), (size_t)m * n * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PREFETCH(d_C, (size_t)m * n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_d, d_f.data(), (size_t)m * sizeof(float), cudaMemcpyHostToDevice));
    GPU_PREFETCH(d_d, (size_t)m * sizeof(float));

    // --- Query cuSOLVER workspace for max sub-problem size (m x n) ---
    int lwork_geqrf = 0, lwork_ormqr = 0;
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(solver_handle, m, n, d_C_sub, m, &lwork_geqrf));
    CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                                m, 1, n, d_C_sub, m, d_tau,
                                                d_d_work, m, &lwork_ormqr));
    int lwork = max(lwork_geqrf, lwork_ormqr);
    GPU_ALLOC(&d_work, (size_t)lwork * sizeof(float));

    // --- Host-side state ---
    vector<float> x(n, 0.0f);
    vector<bool> activeSet(n, false);
    vector<float> gradient(n);
    vector<float> h_R((size_t)m * n);   // host buffer for QR R factor
    vector<float> h_rhs(m);             // host buffer for Q^T * d

    auto start_time = chrono::high_resolution_clock::now();

    // === Main Active Set loop ===
    for (int outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {
        // Upload x to GPU
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        // r = d (D2D copy)
        CUDA_CHECK(cudaMemcpy(d_r, d_d, (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice));

        // r = d - C*x  (r = -1*C*x + 1*r)
        float neg_one = -1.0f, one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, m, n,
                                 &neg_one, d_C, m, d_x, 1, &one, d_r, 1));

        // g = C^T * r
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, m, n,
                                 &one, d_C, m, d_r, 1, &zero, d_g, 1));

        // Download gradient
        CUDA_CHECK(cudaMemcpy(gradient.data(), d_g, n * sizeof(float), cudaMemcpyDeviceToHost));

        // Find max gradient among passive variables
        int j_max = -1;
        float max_gradient = -numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            if (!activeSet[j] && gradient[j] > max_gradient) {
                max_gradient = gradient[j];
                j_max = j;
            }
        }

        if (max_gradient < tol) break;

        activeSet[j_max] = true;

        // === Inner loop: solve sub-problem, remove negatives ===
        for (int inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {
            vector<int> active_idx;
            for (int j = 0; j < n; ++j)
                if (activeSet[j]) active_idx.push_back(j);
            int np = (int)active_idx.size();
            if (np == 0) break;

            // Extract active columns via D2D copies
            for (int i = 0; i < np; ++i) {
                int j = active_idx[i];
                CUDA_CHECK(cudaMemcpy(d_C_sub + (size_t)i * m, d_C + (size_t)j * m,
                                     m * sizeof(float), cudaMemcpyDeviceToDevice));
            }

            // Fresh copy of d
            CUDA_CHECK(cudaMemcpy(d_d_work, d_d, (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice));

            // QR factorize: d_C_sub (m x np)
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
            vector<float> x_active(np);
            for (int i = np - 1; i >= 0; --i) {
                float sum = h_rhs[i];
                for (int j = i + 1; j < np; ++j)
                    sum -= h_R[i + (size_t)j * m] * x_active[j];
                x_active[i] = (fabsf(h_R[i + (size_t)i * m]) > 1e-7f)
                    ? sum / h_R[i + (size_t)i * m] : 0.0f;
            }

            // Map back to full x
            size_t idx = 0;
            for (int j = 0; j < n; ++j) {
                if (activeSet[j])
                    x[j] = x_active[idx++];
                else
                    x[j] = 0.0f;
            }

            // Check for negatives and remove from active set
            bool all_nonneg = true;
            for (int j = 0; j < n; ++j) {
                if (activeSet[j] && x[j] < 0) {
                    all_nonneg = false;
                    activeSet[j] = false;
                }
            }
            if (all_nonneg) break;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

    // --- Cleanup GPU ---
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_C_sub));
    CUDA_CHECK(cudaFree(d_d_work));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    // --- Output (convert float -> double for MATLAB) ---
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (int i = 0; i < n; ++i) out[i] = (double)x[i];

    mexPrintf("NNLS Active Set (FP32, CUDA) - Execution Time: %lld microseconds\n",
             duration.count());
}
