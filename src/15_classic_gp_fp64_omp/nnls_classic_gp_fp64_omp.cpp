/**
 * @file nnls_classic_gp_fp64_omp.cpp
 * @brief Plain projected gradient for NNLS (FP64, OpenMP/BLAS)
 *
 * Iterates  x_{k+1} = max(0, x_k - (1/L) * C^T (C x_k - d))
 * with L estimated by power iteration on C^T C (no Gram formed).
 *
 * Reference baseline (Goldstein-Levitin-Polyak 1964). Asymptotic rate
 * (1 - mu/L) per iter — for Tikhonov PSF problems with kappa ~ 1e10
 * this is ~4e-11, so the algorithm cannot reach low solution error in
 * any practical iter budget. Included as the textbook reference point.
 *
 * @license MIT License
 */

#include "mex.h"
#include "blas.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3)
        mexErrMsgIdAndTxt("nnls_classic_gp_fp64_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_classic_gp_fp64_omp:input",
            "C and d must be real double precision");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_classic_gp_fp64_omp:input",
            "d must be a column vector of length size(C,1)");

    const double* C = mxGetPr(mxC);
    const double* d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    const int max_iter = 500;

    mexPrintf("[GP] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n"
              "     Plain projected gradient on Tikhonov PSF problems (kappa ~1e10):\n"
              "       relErr 0.05  needs ~6.7e10 iterations  (>>years on any hardware)\n"
              "       relErr 0.01  needs ~1.0e11 iterations  (>>years on any hardware)\n"
              "     This algorithm class cannot reach low error on these problems.\n"
              "     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n",
              max_iter);

    const char trans_N = 'N';
    const char trans_T = 'T';
    const double f_one  = 1.0;
    const double f_neg_one = -1.0;
    const double f_zero = 0.0;
    const ptrdiff_t inc1 = 1;

    auto start = chrono::high_resolution_clock::now();

    // Power iteration on C for L = ||C^T C||_2
    vector<double> v((size_t)n), Cv((size_t)m), Hv((size_t)n);
    {
        mt19937 rng(0);
        normal_distribution<double> dist(0.0, 1.0);
        double s2 = 0.0;
        for (ptrdiff_t j = 0; j < n; ++j) { v[(size_t)j] = dist(rng); s2 += v[(size_t)j]*v[(size_t)j]; }
        double s = sqrt(s2);
        if (s > 0.0) { double inv = 1.0 / s; for (ptrdiff_t j = 0; j < n; ++j) v[(size_t)j] *= inv; }
    }
    double L = 0.0;
    for (int k = 0; k < 30; ++k) {
        dgemv(&trans_N, &m, &n, &f_one, C, &m, v.data(), &inc1, &f_zero, Cv.data(), &inc1);
        dgemv(&trans_T, &m, &n, &f_one, C, &m, Cv.data(), &inc1, &f_zero, Hv.data(), &inc1);
        double L_new = sqrt(ddot(&n, Hv.data(), &inc1, Hv.data(), &inc1));
        if (L_new <= 0.0) { L_new = 1.0; L = L_new; break; }
        double inv = 1.0 / L_new;
        for (ptrdiff_t j = 0; j < n; ++j) v[(size_t)j] = Hv[(size_t)j] * inv;
        if (k > 0 && fabs(L_new - L) < 1e-4 * L_new) { L = L_new; break; }
        L = L_new;
    }
    L *= 1.01;
    const double inv_L = 1.0 / L;

    vector<double> x((size_t)n, 0.0);
    vector<double> r((size_t)m, 0.0);
    vector<double> g((size_t)n, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        // r = C*x - d
        memcpy(r.data(), d, (size_t)m * sizeof(double));
        dgemv(&trans_N, &m, &n, &f_one, C, &m, x.data(), &inc1, &f_neg_one, r.data(), &inc1);

        // g = C^T * r
        dgemv(&trans_T, &m, &n, &f_one, C, &m, r.data(), &inc1, &f_zero, g.data(), &inc1);

        // x = max(0, x - inv_L * g)
        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            double xn = x[(size_t)j] - inv_L * g[(size_t)j];
            if (xn < 0.0) xn = 0.0;
            x[(size_t)j] = xn;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    memcpy(mxGetPr(plhs[0]), x.data(), (size_t)n * sizeof(double));

    mexPrintf("NNLS Classic GP (FP64, BLAS, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
