/**
 * @file nnls_cg_fp64_omp.cpp
 * @brief Plain conjugate gradient on B'B x = B'd + terminal projection (FP64, OpenMP/BLAS)
 *
 * Solves the normal equations B'B x = B'd via Hestenes-Stiefel conjugate
 * gradient, then clamps to x >= 0 with a single max(0, .) projection at
 * the end. The bound is not enforced during iteration.
 *
 * Reference baseline for the cost of a gradient-only Krylov method on
 * PSF-class Tikhonov problems where kappa ~ 10^10. Plain CG needs O(10^5)
 * iterations to reach low solution error on this class.
 *
 * Per iter: 2 gemvs on B (one B*p, one B^T * (B*p)).
 *
 * @license MIT License
 */

#include "mex.h"
#include "blas.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3)
        mexErrMsgIdAndTxt("nnls_cg_fp64_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_cg_fp64_omp:input",
            "C and d must be real double precision");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_cg_fp64_omp:input",
            "d must be a column vector of length size(C,1)");

    const double* C = mxGetPr(mxC);
    const double* d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    const int max_iter = 500;

    mexPrintf("[CG] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n"
              "     Plain CG on Tikhonov PSF problems (kappa ~1e10):\n"
              "       relErr 0.05  needs ~55,000 iterations  (~8 min on CUDA FP32)\n"
              "       relErr 0.01  needs ~85,000 iterations  (~12 min on CUDA FP32)\n"
              "     With max_iter=%d this run will return relErr ~0.97.\n"
              "     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n",
              max_iter, max_iter);

    const char trans_N = 'N';
    const char trans_T = 'T';
    const double f_one  = 1.0;
    const double f_zero = 0.0;
    const ptrdiff_t inc1 = 1;

    auto start = chrono::high_resolution_clock::now();

    // x = 0;  g = B'd  (= residual of normal eq at x=0)
    vector<double> x((size_t)n, 0.0);
    vector<double> g((size_t)n, 0.0);
    vector<double> p((size_t)n, 0.0);
    vector<double> Bp((size_t)m, 0.0);
    vector<double> Hp((size_t)n, 0.0);

    // g = C^T * d
    dgemv(&trans_T, &m, &n, &f_one, C, &m,
          d, &inc1, &f_zero, g.data(), &inc1);

    // p = g; rs_old = g'*g
    memcpy(p.data(), g.data(), (size_t)n * sizeof(double));
    double rs_old = 0.0;
    #pragma omp parallel for reduction(+:rs_old) schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j) rs_old += g[(size_t)j] * g[(size_t)j];

    for (int iter = 0; iter < max_iter; ++iter) {
        // Bp = C * p
        dgemv(&trans_N, &m, &n, &f_one, C, &m,
              p.data(), &inc1, &f_zero, Bp.data(), &inc1);

        // Hp = C^T * Bp
        dgemv(&trans_T, &m, &n, &f_one, C, &m,
              Bp.data(), &inc1, &f_zero, Hp.data(), &inc1);

        // denom = p' * Hp
        double denom = 0.0;
        #pragma omp parallel for reduction(+:denom) schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) denom += p[(size_t)j] * Hp[(size_t)j];
        if (denom <= 0.0) break;

        double alpha = rs_old / denom;

        // x += alpha * p;  g -= alpha * Hp
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            x[(size_t)j] += alpha * p[(size_t)j];
            g[(size_t)j] -= alpha * Hp[(size_t)j];
        }

        // rs_new = g'*g
        double rs_new = 0.0;
        #pragma omp parallel for reduction(+:rs_new) schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) rs_new += g[(size_t)j] * g[(size_t)j];
        if (rs_new <= 0.0) break;

        double beta = rs_new / rs_old;

        // p = g + beta * p
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            p[(size_t)j] = g[(size_t)j] + beta * p[(size_t)j];
        }
        rs_old = rs_new;
    }

    // Terminal projection
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j) if (x[(size_t)j] < 0.0) x[(size_t)j] = 0.0;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    memcpy(mxGetPr(plhs[0]), x.data(), (size_t)n * sizeof(double));

    mexPrintf("NNLS CG (FP64, BLAS, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
