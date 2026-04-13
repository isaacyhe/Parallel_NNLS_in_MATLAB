/**
 * @file nnls_cg_fp32_omp.cpp
 * @brief Plain conjugate gradient on B'B x = B'd + terminal projection (FP32, OpenMP/BLAS)
 *
 * Single-precision plain CG. See nnls_cg_fp64_omp.cpp for the algorithm
 * and rationale.
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
        mexErrMsgIdAndTxt("nnls_cg_fp32_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_cg_fp32_omp:input", "C and d must be real");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_cg_fp32_omp:input",
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
    const float f_one  = 1.0f;
    const float f_zero = 0.0f;
    const ptrdiff_t inc1 = 1;

    auto start = chrono::high_resolution_clock::now();

    vector<float> x((size_t)n, 0.0f);
    vector<float> g((size_t)n, 0.0f);
    vector<float> p((size_t)n, 0.0f);
    vector<float> Bp((size_t)m, 0.0f);
    vector<float> Hp((size_t)n, 0.0f);

    // g = C^T * d
    sgemv(&trans_T, &m, &n, &f_one, C32.data(), &m,
          d32.data(), &inc1, &f_zero, g.data(), &inc1);

    memcpy(p.data(), g.data(), (size_t)n * sizeof(float));
    float rs_old = 0.0f;
    #pragma omp parallel for reduction(+:rs_old) schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j) rs_old += g[(size_t)j] * g[(size_t)j];

    for (int iter = 0; iter < max_iter; ++iter) {
        sgemv(&trans_N, &m, &n, &f_one, C32.data(), &m,
              p.data(), &inc1, &f_zero, Bp.data(), &inc1);

        sgemv(&trans_T, &m, &n, &f_one, C32.data(), &m,
              Bp.data(), &inc1, &f_zero, Hp.data(), &inc1);

        float denom = 0.0f;
        #pragma omp parallel for reduction(+:denom) schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) denom += p[(size_t)j] * Hp[(size_t)j];
        if (denom <= 0.0f) break;

        float alpha = rs_old / denom;

        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            x[(size_t)j] += alpha * p[(size_t)j];
            g[(size_t)j] -= alpha * Hp[(size_t)j];
        }

        float rs_new = 0.0f;
        #pragma omp parallel for reduction(+:rs_new) schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) rs_new += g[(size_t)j] * g[(size_t)j];
        if (rs_new <= 0.0f) break;

        float beta = rs_new / rs_old;

        #pragma omp parallel for schedule(static)
        for (ptrdiff_t j = 0; j < n; ++j) {
            p[(size_t)j] = g[(size_t)j] + beta * p[(size_t)j];
        }
        rs_old = rs_new;
    }

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j) if (x[(size_t)j] < 0.0f) x[(size_t)j] = 0.0f;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Output as double for caller compatibility
    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (ptrdiff_t j = 0; j < n; ++j) out[(size_t)j] = (double)x[(size_t)j];

    mexPrintf("NNLS CG (FP32, BLAS, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
