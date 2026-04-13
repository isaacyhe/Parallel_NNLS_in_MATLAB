/**
 * @file nnls_admm_fp32_omp.cpp
 * @brief ADMM for NNLS, BLAS/LAPACK/OpenMP (FP32)
 *
 * Solves  min ||C*x - d||^2  subject to  x >= 0  via ADMM.
 *
 * Splitting: min ||C*x - d||^2 + I_+(z)  s.t.  x = z
 * Updates (scaled form):
 *     x <- (2*C'*C + rho*I)^{-1} * (2*C'*d + rho*(z - u))
 *     z <- max(0, x + u)
 *     u <- u + (x - z)
 *
 * The x-update is reduced to two triangular solves after a one-time
 * Cholesky factorization of M = 2*C'*C + rho*I. Convergence is independent
 * of cond(C), so this works on Tikhonov problems where FISTA dies.
 *
 * Single precision: ssyrk + spotrf + spotrs.
 *
 * @license MIT License
 */

#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3)
        mexErrMsgIdAndTxt("nnls_admm_fp32_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_admm_fp32_omp:input", "C and d must be real");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_admm_fp32_omp:input",
            "d must be a column vector of length size(C,1)");

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    // Convert input to float
    vector<float> C32((size_t)m * (size_t)n);
    vector<float> d32((size_t)m);
    {
        const double* Cd = mxGetPr(mxC);
        const double* dd = mxGetPr(mxD);
        size_t total = (size_t)m * (size_t)n;
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)total; ++i) C32[(size_t)i] = (float)Cd[(size_t)i];
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < m; ++i) d32[(size_t)i] = (float)dd[(size_t)i];
    }

    const float rho      = 10.0f;
    const int   max_iter = 500;

    const char uplo_U  = 'U';
    const char trans_T = 'T';
    const float f_one  = 1.0f;
    const float f_zero = 0.0f;
    const ptrdiff_t inc1 = 1;
    const ptrdiff_t nrhs1 = 1;

    auto start = chrono::high_resolution_clock::now();

    // ---- Pre-compute Gram H = C^T * C (upper triangle only) ----
    vector<float> H((size_t)n * (size_t)n, 0.0f);
    ssyrk(&uplo_U, &trans_T, &n, &m, &f_one,
          C32.data(), &m, &f_zero, H.data(), &n);

    // ---- Pre-compute q = C^T * d ----
    vector<float> q((size_t)n, 0.0f);
    sgemv(&trans_T, &m, &n, &f_one, C32.data(), &m,
          d32.data(), &inc1, &f_zero, q.data(), &inc1);

    // Free input copies (no longer needed)
    C32 = vector<float>(); d32 = vector<float>();

    // ---- Build M = 2*H + rho*I  (upper triangle) ----
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j) {
        float* col = H.data() + (size_t)j * (size_t)n;
        for (ptrdiff_t i = 0; i <= j; ++i) col[(size_t)i] *= 2.0f;
        col[(size_t)j] += rho;
    }

    // ---- Cholesky factor: spotrf ----
    ptrdiff_t info = 0;
    spotrf(&uplo_U, &n, H.data(), &n, &info);
    if (info != 0)
        mexErrMsgIdAndTxt("nnls_admm_fp32_omp:chol",
            "spotrf failed (info=%lld)", (long long)info);

    // ---- ADMM iterations ----
    vector<float> x((size_t)n, 0.0f);
    vector<float> z((size_t)n, 0.0f);
    vector<float> u((size_t)n, 0.0f);
    vector<float> rhs((size_t)n, 0.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            rhs[(size_t)j] = 2.0f * q[(size_t)j] + rho * (z[(size_t)j] - u[(size_t)j]);
        }

        memcpy(x.data(), rhs.data(), (size_t)n * sizeof(float));
        spotrs(&uplo_U, &n, &nrhs1, H.data(), &n, x.data(), &n, &info);

        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            float xj = x[(size_t)j];
            float uj = u[(size_t)j];
            float zn = xj + uj;
            if (zn < 0.0f) zn = 0.0f;
            u[(size_t)j] = uj + xj - zn;
            z[(size_t)j] = zn;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Output as double for caller compatibility
    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (ptrdiff_t j = 0; j < n; ++j) out[(size_t)j] = (double)z[(size_t)j];

    mexPrintf("NNLS ADMM (FP32, BLAS/LAPACK, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
