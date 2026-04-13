/**
 * @file nnls_fast_nnls_fp32_omp.cpp
 * @brief FAST-NNLS (Cobb et al., IEEE BigData 2025) with OpenMP + BLAS (FP32)
 *
 * Single-precision variant of the incremental-Cholesky FAST-NNLS solver.
 * Threshold-based batch active-set method. Like FNNLS but adds/removes
 * multiple variables per iteration using threshold parameters theta_add
 * and theta_rem for faster convergence on large problems.
 *
 * This implementation does NOT pre-compute the n x n Gram matrix ZtZ = C'*C
 * (infeasible for n in the tens of thousands). Also, FP32 cannot tolerate
 * the normal-equations approach (condition^2), so we factor C_act directly
 * via QR (sgeqrf + sormqr + strsv). Each time the active set changes,
 * we rebuild from scratch:
 *   - Copy active columns into C_act_qr (m x p)
 *   - sgeqrf: R in upper triangle, Householder reflectors below diag, tau
 *   - sormqr: apply Q^T to a fresh copy of d -> qtd
 *   - strsv on R solves the sub-problem
 * Gradient via two sgemv calls on original C:
 *     r = d - C*x
 *     w = C^T * r
 *
 * References:
 *   - Cobb et al. (2025). FAST-NNLS: A fast and exact non-negative least
 *     squares algorithm. IEEE BigData.
 *   - Lawson, C. L., & Hanson, R. J. (1995). Solving Least Squares Problems.
 *
 * @license MIT License
 */

#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3)
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp32_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp32_omp:input",
            "C and d must be real double precision (they will be cast to float)");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp32_omp:input",
            "d must be a column vector of length size(C,1)");

    const double* C_d = mxGetPr(mxC);
    const double* d_d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    // Cast input to single precision
    vector<float> C((size_t)m * (size_t)n);
    vector<float> d((size_t)m);
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t j = 0; j < n; ++j)
        for (ptrdiff_t i = 0; i < m; ++i)
            C[(size_t)j * (size_t)m + (size_t)i] = (float)C_d[i + j * m];
    for (ptrdiff_t i = 0; i < m; ++i)
        d[(size_t)i] = (float)d_d[i];

    const float tol = 1e-5f;
    const float feas_tol = 1e-5f;  // FP32 noise floor for feasibility
    const int max_iter = 3 * (int)n;
    const float theta_add = 0.5f;
    const float theta_rem = 0.5f;

    // BLAS/LAPACK constants
    const char uplo_U = 'U';
    const char trans_N = 'N';
    const char trans_T = 'T';
    const char diag_N = 'N';
    const float f_one = 1.0f;
    const float f_neg_one = -1.0f;
    const float f_zero = 0.0f;
    const ptrdiff_t inc1 = 1;
    const ptrdiff_t nld = n;

    // Solver state
    vector<float> x((size_t)n, 0.0f);
    vector<float> s((size_t)n, 0.0f);
    vector<float> w((size_t)n, 0.0f);   // w = C'*(d - C*x)
    vector<float> r((size_t)m, 0.0f);
    vector<unsigned char> P((size_t)n, 0);

    // Sub-problem factorization state
    vector<float> C_act((size_t)m * (size_t)n);
    vector<float> tau((size_t)n, 0.0f);
    vector<float> qtd((size_t)m, 0.0f);  // Q^T * d
    vector<float> s_sub((size_t)n, 0.0f);
    vector<ptrdiff_t> idx_P;
    idx_P.reserve((size_t)n);
    ptrdiff_t p = 0;

    // Workspace for sgeqrf / sormqr: query once with max possible size.
    ptrdiff_t lwork = -1;
    float wopt = 0.0f;
    ptrdiff_t info_q = 0;
    sgeqrf(&m, &n, C_act.data(), &m, tau.data(), &wopt, &lwork, &info_q);
    ptrdiff_t lwork_geqrf = (ptrdiff_t)wopt;

    const char side_L = 'L';
    ptrdiff_t one_rhs = 1;
    lwork = -1; wopt = 0.0f; info_q = 0;
    sormqr(&side_L, &trans_T, &m, &one_rhs, &n, C_act.data(), &m,
           tau.data(), qtd.data(), &m, &wopt, &lwork, &info_q);
    ptrdiff_t lwork_ormqr = (ptrdiff_t)wopt;

    ptrdiff_t lwork_max = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
    if (lwork_max < 1) lwork_max = 1;
    vector<float> work((size_t)lwork_max, 0.0f);

    auto sync_idx_P = [&]() {
        idx_P.clear();
        for (ptrdiff_t j = 0; j < n; ++j)
            if (P[(size_t)j]) idx_P.push_back(j);
        p = (ptrdiff_t)idx_P.size();
    };

    auto rebuild = [&]() -> int {
        if (p == 0) return 0;

        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < p; ++i) {
            memcpy(C_act.data() + (size_t)i * (size_t)m,
                   C.data() + (size_t)idx_P[(size_t)i] * (size_t)m,
                   (size_t)m * sizeof(float));
        }

        // QR factorization of m x p active sub-matrix (in place).
        ptrdiff_t info = 0;
        ptrdiff_t lw = (ptrdiff_t)work.size();
        sgeqrf(&m, &p, C_act.data(), &m, tau.data(), work.data(), &lw, &info);
        if (info != 0) return (int)info;

        // Apply Q^T to d: qtd = Q^T * d
        memcpy(qtd.data(), d.data(), (size_t)m * sizeof(float));
        info = 0;
        lw = (ptrdiff_t)work.size();
        sormqr(&side_L, &trans_T, &m, &one_rhs, &p, C_act.data(), &m,
               tau.data(), qtd.data(), &m, work.data(), &lw, &info);
        if (info != 0) return (int)info;
        return 0;
    };

    auto solve_LS = [&]() {
        // s_sub = R^{-1} * qtd(1:p); R is upper triangle of C_act, leading m x p
        memcpy(s_sub.data(), qtd.data(), (size_t)p * sizeof(float));
        strsv(&uplo_U, &trans_N, &diag_N, &p, C_act.data(), &m,
              s_sub.data(), &inc1);
    };

    auto start = chrono::high_resolution_clock::now();

    // Initial gradient w = C' * d  (x = 0)
    sgemv(&trans_T, &m, &n, &f_one, C.data(), &m,
          d.data(), &inc1, &f_zero, w.data(), &inc1);

    int iter = 0;
    while (true) {
        // Find max w over non-passive set
        float max_w = -numeric_limits<float>::infinity();
        for (ptrdiff_t j = 0; j < n; ++j)
            if (!P[(size_t)j] && w[(size_t)j] > max_w) max_w = w[(size_t)j];
        if (max_w <= tol) break;

        // BATCH ADD: all j not in P with w[j] > theta_add * max_w
        float t_add = max_w * theta_add;
        bool added = false;
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (!P[(size_t)j] && w[(size_t)j] > t_add) {
                P[(size_t)j] = 1;
                added = true;
            }
        }
        if (!added) break;

        sync_idx_P();
        if (rebuild() != 0) break;

        // Inner feasibility loop
        while (true) {
            iter++;
            if (iter > max_iter) break;
            if (p == 0) break;

            solve_LS();

            // Scatter s_sub into full s
            for (ptrdiff_t j = 0; j < n; ++j) s[(size_t)j] = 0.0f;
            for (ptrdiff_t i = 0; i < p; ++i)
                s[(size_t)idx_P[(size_t)i]] = s_sub[(size_t)i];

            // Feasibility check (with FP32 noise tolerance)
            bool has_infeasible = false;
            for (ptrdiff_t i = 0; i < p; ++i) {
                if (s_sub[(size_t)i] < -feas_tol) { has_infeasible = true; break; }
            }
            if (!has_infeasible) break;

            // BATCH REMOVE
            float min_s = numeric_limits<float>::infinity();
            for (ptrdiff_t i = 0; i < p; ++i)
                if (s_sub[(size_t)i] < min_s) min_s = s_sub[(size_t)i];
            float t_rem = min_s * theta_rem;

            // Interpolation alpha: x += alpha * (s - x)
            float alpha = numeric_limits<float>::infinity();
            for (ptrdiff_t i = 0; i < p; ++i) {
                ptrdiff_t j = idx_P[(size_t)i];
                if (s[(size_t)j] < t_rem) {
                    float denom = x[(size_t)j] - s[(size_t)j];
                    if (denom > 0.0f) {
                        float a = x[(size_t)j] / denom;
                        if (a < alpha) alpha = a;
                    }
                }
            }
            if (!isfinite(alpha)) alpha = 0.0f;

            // Degenerate case: alpha == 0 means x=0 or no progress possible.
            // Fall back to projection: x = max(s, 0) for passive, then kick out zeros.
            ptrdiff_t p_before = p;
            if (alpha <= 0.0f) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (P[(size_t)j]) {
                        float sj = s[(size_t)j];
                        if (sj > feas_tol) {
                            x[(size_t)j] = sj;
                        } else {
                            x[(size_t)j] = 0.0f;
                            P[(size_t)j] = 0;
                        }
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j)
                    x[(size_t)j] += alpha * (s[(size_t)j] - x[(size_t)j]);

                // Remove near-zero passive vars where s was non-positive
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (P[(size_t)j] && fabsf(x[(size_t)j]) < feas_tol && s[(size_t)j] <= feas_tol) {
                        P[(size_t)j] = 0;
                        x[(size_t)j] = 0.0f;
                    }
                }
            }

            sync_idx_P();
            // Safety: if nothing actually got removed, break to avoid infinite loop
            if (p == p_before) break;
            if (rebuild() != 0) { iter = max_iter + 1; break; }
        }

        if (iter > max_iter) break;

        // Accept s as x
        memcpy(x.data(), s.data(), (size_t)n * sizeof(float));

        // Update gradient: r = d - C*x;  w = C^T * r
        memcpy(r.data(), d.data(), (size_t)m * sizeof(float));
        sgemv(&trans_N, &m, &n, &f_neg_one, C.data(), &m,
              x.data(), &inc1, &f_one, r.data(), &inc1);
        sgemv(&trans_T, &m, &n, &f_one, C.data(), &m,
              r.data(), &inc1, &f_zero, w.data(), &inc1);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Ensure non-negativity
    for (ptrdiff_t i = 0; i < n; ++i)
        if (x[(size_t)i] < 0.0f) x[(size_t)i] = 0.0f;

    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (ptrdiff_t i = 0; i < n; ++i) out[i] = (double)x[(size_t)i];

    mexPrintf("NNLS FAST-NNLS (FP32, incremental Cholesky, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
