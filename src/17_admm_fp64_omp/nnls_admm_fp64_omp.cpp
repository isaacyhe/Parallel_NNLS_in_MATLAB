/**
 * @file nnls_admm_fp64_omp.cpp
 * @brief ADMM for NNLS, BLAS/LAPACK/OpenMP (FP64)
 *
 * Solves  min ||C*x - d||^2  subject to  x >= 0
 * via ADMM with the splitting:
 *     min ||C*x - d||^2 + I_+(z)   s.t.  x = z
 *
 * Updates (scaled form):
 *     x <- (2*C'*C + rho*I)^{-1} * (2*C'*d + rho*(z - u))
 *     z <- max(0, x + u)
 *     u <- u + (x - z)
 *
 * The x-update is reduced to two triangular solves after a one-time
 * Cholesky factorization of M = 2*C'*C + rho*I. Convergence is
 * independent of the conditioning of C — what kills FISTA on Tikhonov
 * NNLS (kappa ~ 10^10) is irrelevant here.
 *
 * Implementation:
 *   - dsyrk forms the upper triangle of H = C^T*C
 *   - dgemv computes q = C^T*d
 *   - Custom OMP loop builds 2*H + rho*I (upper triangle) in place
 *   - dpotrf factors M = R^T*R (upper)
 *   - Per iter: dpotrs solves M*x = rhs in one call (two trsv internally)
 *
 * Reference:
 *   Boyd, Parikh, Chu, Peleato & Eckstein (2011). Distributed optimization
 *     and statistical learning via ADMM. Foundations and Trends in ML.
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
        mexErrMsgIdAndTxt("nnls_admm_fp64_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_admm_fp64_omp:input",
            "C and d must be real double precision");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_admm_fp64_omp:input",
            "d must be a column vector of length size(C,1)");

    const double* C = mxGetPr(mxC);
    const double* d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    const double rho      = 10.0;
    const int    max_iter = 500;

    // BLAS/LAPACK constants
    const char uplo_U  = 'U';
    const char trans_T = 'T';
    const double f_one  = 1.0;
    const double f_zero = 0.0;
    const ptrdiff_t inc1 = 1;
    const ptrdiff_t nrhs1 = 1;

    auto start = chrono::high_resolution_clock::now();

    // ---- Pre-compute Gram H = C^T * C (upper triangle only) ----
    vector<double> H((size_t)n * (size_t)n, 0.0);
    dsyrk(&uplo_U, &trans_T, &n, &m, &f_one,
          C, &m, &f_zero, H.data(), &n);

    // ---- Pre-compute q = C^T * d ----
    vector<double> q((size_t)n, 0.0);
    dgemv(&trans_T, &m, &n, &f_one, C, &m,
          d, &inc1, &f_zero, q.data(), &inc1);

    // ---- Build M = 2*H + rho*I  (upper triangle) ----
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j) {
        double* col = H.data() + (size_t)j * (size_t)n;
        for (ptrdiff_t i = 0; i <= j; ++i) col[(size_t)i] *= 2.0;
        col[(size_t)j] += rho;
    }

    // ---- Cholesky factor: dpotrf  (M = R^T R, upper) ----
    ptrdiff_t info = 0;
    dpotrf(&uplo_U, &n, H.data(), &n, &info);
    if (info != 0)
        mexErrMsgIdAndTxt("nnls_admm_fp64_omp:chol",
            "dpotrf failed (info=%lld)", (long long)info);

    // ---- ADMM iterations ----
    vector<double> x((size_t)n, 0.0);
    vector<double> z((size_t)n, 0.0);
    vector<double> u((size_t)n, 0.0);
    vector<double> rhs((size_t)n, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        // rhs = 2*q + rho*(z - u)
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            rhs[(size_t)j] = 2.0 * q[(size_t)j] + rho * (z[(size_t)j] - u[(size_t)j]);
        }

        // x = M^{-1} rhs via dpotrs (two trsv inside)
        memcpy(x.data(), rhs.data(), (size_t)n * sizeof(double));
        dpotrs(&uplo_U, &n, &nrhs1, H.data(), &n, x.data(), &n, &info);

        // z = max(0, x + u);  u += x - z
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            double xj = x[(size_t)j];
            double uj = u[(size_t)j];
            double zn = xj + uj;
            if (zn < 0.0) zn = 0.0;
            u[(size_t)j] = uj + xj - zn;
            z[(size_t)j] = zn;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Return z (non-negative by construction)
    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    memcpy(mxGetPr(plhs[0]), z.data(), (size_t)n * sizeof(double));

    mexPrintf("NNLS ADMM (FP64, BLAS/LAPACK, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
