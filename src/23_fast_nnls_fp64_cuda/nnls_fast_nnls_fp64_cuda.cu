/**
 * @file nnls_fast_nnls_fp64_cuda.cu
 * @brief FAST-NNLS (Cobb et al., IEEE BigData 2025) with CUDA (FP64)
 *
 * GPU-resident implementation: C and d stay on GPU throughout. cuBLAS for
 * pre-computation (ZtZ, Zty) and gradient updates. cuSOLVER for QR
 * factorization of sub-problems. No per-iteration alloc/free.
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
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp64_cuda:input",
            "Two inputs required: C, d");

    int m = (int)mxGetM(prhs[0]);
    int n = (int)mxGetN(prhs[0]);
    const double* C = mxGetPr(prhs[0]);   // column-major from MATLAB
    const double* d = mxGetPr(prhs[1]);

    double tol = 1e-8;
    int max_iter = 30 * n;
    double theta_add = 0.5;
    double theta_rem = 0.5;

    // --- Create handles ---
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t solver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

    // --- Allocate persistent GPU buffers ---
    double *d_C, *d_d, *d_ZtZ, *d_Zty;
    double *d_x, *d_w;
    double *d_C_sub, *d_d_work, *d_tau, *d_work;
    int *d_info;

    GPU_ALLOC(&d_C,     (size_t)m * n * sizeof(double));
    GPU_ALLOC(&d_d,     (size_t)m * sizeof(double));
    GPU_ALLOC(&d_ZtZ,   (size_t)n * n * sizeof(double));
    GPU_ALLOC(&d_Zty,   (size_t)n * sizeof(double));
    GPU_ALLOC(&d_x,     (size_t)n * sizeof(double));
    GPU_ALLOC(&d_w,     (size_t)n * sizeof(double));
    GPU_ALLOC(&d_C_sub, (size_t)m * n * sizeof(double));
    GPU_ALLOC(&d_d_work,(size_t)m * sizeof(double));
    GPU_ALLOC(&d_tau,   (size_t)n * sizeof(double));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // --- Upload data ---
    CUDA_CHECK(cudaMemcpy(d_C, C, (size_t)m * n * sizeof(double), cudaMemcpyHostToDevice));
    GPU_PREFETCH(d_C, (size_t)m * n * sizeof(double));
    CUDA_CHECK(cudaMemcpy(d_d, d, (size_t)m * sizeof(double), cudaMemcpyHostToDevice));
    GPU_PREFETCH(d_d, (size_t)m * sizeof(double));

    // --- Pre-compute ZtZ = C'*C and Zty = C'*d on GPU ---
    double alpha_blas = 1.0, beta_blas = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &alpha_blas, d_C, m, d_C, m, &beta_blas, d_ZtZ, n));
    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_T, m, n,
                             &alpha_blas, d_C, m, d_d, 1, &beta_blas, d_Zty, 1));

    // Download Zty for initial w
    vector<double> Zty(n);
    CUDA_CHECK(cudaMemcpy(Zty.data(), d_Zty, n * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Query cuSOLVER workspace for max sub-problem size (m x n) ---
    int lwork_geqrf = 0, lwork_ormqr = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(solver_handle, m, n, d_C_sub, m, &lwork_geqrf));
    CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                                m, 1, n, d_C_sub, m, d_tau,
                                                d_d_work, m, &lwork_ormqr));
    int lwork = max(lwork_geqrf, lwork_ormqr);
    GPU_ALLOC(&d_work, (size_t)lwork * sizeof(double));

    auto start = chrono::high_resolution_clock::now();

    // === FAST-NNLS main algorithm ===
    vector<double> x(n, 0.0);
    vector<bool> P(n, false);
    vector<double> w(Zty);
    vector<double> s(n, 0.0);

    int iter = 0;

    while (true) {
        double max_w = -numeric_limits<double>::infinity();
        for (int j = 0; j < n; ++j) {
            if (!P[j] && w[j] > max_w) max_w = w[j];
        }
        if (max_w <= tol) break;

        // BATCH ADD
        double t_add = max_w * theta_add;
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
                                     m * sizeof(double), cudaMemcpyDeviceToDevice));
            }

            // Fresh copy of d
            CUDA_CHECK(cudaMemcpy(d_d_work, d_d, (size_t)m * sizeof(double), cudaMemcpyDeviceToDevice));

            // QR factorize
            CUSOLVER_CHECK(cusolverDnDgeqrf(solver_handle, m, np, d_C_sub, m,
                                            d_tau, d_work, lwork, d_info));

            // Apply Q^T to d_d_work
            CUSOLVER_CHECK(cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                            m, 1, np, d_C_sub, m, d_tau,
                                            d_d_work, m, d_work, lwork, d_info));

            // Solve R * z = (Q^T * d) via triangular solve
            double one_trsm = 1.0;
            CUBLAS_CHECK(cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                     np, 1, &one_trsm, d_C_sub, m, d_d_work, m));

            // Download sub-result
            vector<double> s_sub(np);
            CUDA_CHECK(cudaMemcpy(s_sub.data(), d_d_work, np * sizeof(double),
                                 cudaMemcpyDeviceToHost));

            for (int j = 0; j < n; ++j) s[j] = 0.0;
            for (int i = 0; i < np; ++i) s[idx_P[i]] = s_sub[i];

            bool has_infeasible = false;
            for (int i = 0; i < np; ++i) {
                if (s[idx_P[i]] <= 0.0) { has_infeasible = true; break; }
            }
            if (!has_infeasible) break;

            // BATCH REMOVE
            double min_s = numeric_limits<double>::infinity();
            for (int i = 0; i < np; ++i) {
                if (s[idx_P[i]] < min_s) min_s = s[idx_P[i]];
            }
            double t_rem_val = min_s * theta_rem;

            double alpha = numeric_limits<double>::infinity();
            for (int i = 0; i < np; ++i) {
                int j = idx_P[i];
                if (s[j] < t_rem_val) {
                    double a = x[j] / (x[j] - s[j]);
                    if (a < alpha) alpha = a;
                }
            }

            for (int j = 0; j < n; ++j)
                x[j] += alpha * (s[j] - x[j]);

            for (int j = 0; j < n; ++j) {
                if (P[j] && fabs(x[j]) < tol && s[j] <= 0.0) {
                    P[j] = false;
                    x[j] = 0.0;
                }
            }
        }

        for (int j = 0; j < n; ++j) x[j] = s[j];

        // Gradient update: w = Zty - ZtZ * x  (on GPU)
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, d_Zty, n * sizeof(double), cudaMemcpyDeviceToDevice));
        double neg_one = -1.0, one = 1.0;
        CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, n,
                                 &neg_one, d_ZtZ, n, d_x, 1, &one, d_w, 1));
        CUDA_CHECK(cudaMemcpy(w.data(), d_w, n * sizeof(double), cudaMemcpyDeviceToHost));

        if (iter > max_iter) break;
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    for (int i = 0; i < n; ++i)
        if (x[i] < 0.0) x[i] = 0.0;

    // --- Cleanup GPU ---
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_ZtZ));
    CUDA_CHECK(cudaFree(d_Zty));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_C_sub));
    CUDA_CHECK(cudaFree(d_d_work));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    // --- Output ---
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (int i = 0; i < n; ++i) out[i] = x[i];

    mexPrintf("NNLS FAST-NNLS (FP64, CUDA) - Execution Time: %lld microseconds\n", duration.count());
}
