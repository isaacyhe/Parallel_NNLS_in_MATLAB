/**
 * @file nnls_classic_gp_fp32_omp.cpp
 * @brief Plain projected gradient for NNLS (FP32, OpenMP/BLAS)
 *
 * Single-precision plain GP. See nnls_classic_gp_fp64_omp.cpp for the
 * algorithm and rationale.
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
        mexErrMsgIdAndTxt("nnls_classic_gp_fp32_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_classic_gp_fp32_omp:input", "C and d must be real");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_classic_gp_fp32_omp:input",
            "d must be a column vector of length size(C,1)");

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

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
    const float f_one  = 1.0f;
    const float f_neg_one = -1.0f;
    const float f_zero = 0.0f;
    const ptrdiff_t inc1 = 1;

    auto start = chrono::high_resolution_clock::now();

    // Power iteration on C32 for L
    vector<float> v((size_t)n), Cv((size_t)m), Hv((size_t)n);
    {
        mt19937 rng(0);
        normal_distribution<float> dist(0.0f, 1.0f);
        float s2 = 0.0f;
        for (ptrdiff_t j = 0; j < n; ++j) { v[(size_t)j] = dist(rng); s2 += v[(size_t)j]*v[(size_t)j]; }
        float s = sqrtf(s2);
        if (s > 0.0f) { float inv = 1.0f / s; for (ptrdiff_t j = 0; j < n; ++j) v[(size_t)j] *= inv; }
    }
    float L = 0.0f;
    for (int k = 0; k < 30; ++k) {
        sgemv(&trans_N, &m, &n, &f_one, C32.data(), &m, v.data(), &inc1, &f_zero, Cv.data(), &inc1);
        sgemv(&trans_T, &m, &n, &f_one, C32.data(), &m, Cv.data(), &inc1, &f_zero, Hv.data(), &inc1);
        float L_new = sqrtf(sdot(&n, Hv.data(), &inc1, Hv.data(), &inc1));
        if (L_new <= 0.0f) { L_new = 1.0f; L = L_new; break; }
        float inv = 1.0f / L_new;
        for (ptrdiff_t j = 0; j < n; ++j) v[(size_t)j] = Hv[(size_t)j] * inv;
        if (k > 0 && fabsf(L_new - L) < 1e-4f * L_new) { L = L_new; break; }
        L = L_new;
    }
    L *= 1.01f;
    const float inv_L = 1.0f / L;

    vector<float> x((size_t)n, 0.0f);
    vector<float> r((size_t)m, 0.0f);
    vector<float> g((size_t)n, 0.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        memcpy(r.data(), d32.data(), (size_t)m * sizeof(float));
        sgemv(&trans_N, &m, &n, &f_one, C32.data(), &m, x.data(), &inc1, &f_neg_one, r.data(), &inc1);

        sgemv(&trans_T, &m, &n, &f_one, C32.data(), &m, r.data(), &inc1, &f_zero, g.data(), &inc1);

        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            float xn = x[(size_t)j] - inv_L * g[(size_t)j];
            if (xn < 0.0f) xn = 0.0f;
            x[(size_t)j] = xn;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (ptrdiff_t j = 0; j < n; ++j) out[(size_t)j] = (double)x[(size_t)j];

    mexPrintf("NNLS Classic GP (FP32, BLAS, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
