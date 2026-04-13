/**
 * @file nnls_fast_nnls_fp64_omp.cpp
 * @brief FAST-NNLS (Cobb et al., IEEE BigData 2025) with OpenMP + BLAS (FP64)
 *
 * Threshold-based batch active-set method. Like FNNLS but adds/removes
 * multiple variables per iteration using threshold parameters theta_add
 * and theta_rem for faster convergence on large problems.
 *
 * This implementation does NOT pre-compute the n x n Gram matrix ZtZ = C'*C
 * (infeasible for n in the tens of thousands). Instead it:
 *   - Keeps C in its original m x n form
 *   - Each time the active set changes (add or remove), rebuilds the
 *     Cholesky factor R^T R = C_act^T * C_act via dsyrk + dpotrf (BLAS)
 *   - Solves the sub-problem via two dtrsv calls
 *   - Recomputes the gradient via two dgemv calls on B directly:
 *         r = d - C*x
 *         w = C^T * r
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
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp64_omp:input",
            "Three inputs required: C, d, num_threads");

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD))
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp64_omp:input",
            "C and d must be real double precision");

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1)
        mexErrMsgIdAndTxt("nnls_fast_nnls_fp64_omp:input",
            "d must be a column vector of length size(C,1)");

    const double* C = mxGetPr(mxC);
    const double* d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    const double tol = 1e-8;
    const int max_iter = 3 * (int)n;
    const double theta_add = 0.5;
    const double theta_rem = 0.5;

    // BLAS/LAPACK constants
    const char uplo_U = 'U';
    const char trans_N = 'N';
    const char trans_T = 'T';
    const char diag_N = 'N';
    const double f_one = 1.0;
    const double f_neg_one = -1.0;
    const double f_zero = 0.0;
    const ptrdiff_t inc1 = 1;
    const ptrdiff_t nld = n;

    // Solver state
    vector<double> x((size_t)n, 0.0);
    vector<double> s((size_t)n, 0.0);
    vector<double> w((size_t)n, 0.0);   // w = C'*(d - C*x)
    vector<double> r((size_t)m, 0.0);
    vector<unsigned char> P((size_t)n, 0);

    // Sub-problem factorization state
    vector<double> C_act((size_t)m * (size_t)n);
    vector<double> R((size_t)n * (size_t)n, 0.0);
    vector<double> q_sub((size_t)n, 0.0);
    vector<double> s_sub((size_t)n, 0.0);
    vector<ptrdiff_t> idx_P;
    idx_P.reserve((size_t)n);
    ptrdiff_t p = 0;

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
                   C + (size_t)idx_P[(size_t)i] * (size_t)m,
                   (size_t)m * sizeof(double));
        }

        dsyrk(&uplo_U, &trans_T, &p, &m, &f_one,
              C_act.data(), &m, &f_zero, R.data(), &nld);

        ptrdiff_t info = 0;
        dpotrf(&uplo_U, &p, R.data(), &nld, &info);
        if (info != 0) {
            double trace = 0.0;
            for (ptrdiff_t k = 0; k < p; ++k) {
                const double* col = C_act.data() + (size_t)k * (size_t)m;
                double ss = 0.0;
                for (ptrdiff_t ii = 0; ii < m; ++ii) ss += col[ii] * col[ii];
                trace += ss;
            }
            double jitter = 1e-12 * (trace / (double)p + 1.0);
            dsyrk(&uplo_U, &trans_T, &p, &m, &f_one,
                  C_act.data(), &m, &f_zero, R.data(), &nld);
            for (ptrdiff_t k = 0; k < p; ++k)
                R[(size_t)k * (size_t)nld + (size_t)k] += jitter;
            info = 0;
            dpotrf(&uplo_U, &p, R.data(), &nld, &info);
        }
        if (info != 0) return (int)info;

        dgemv(&trans_T, &m, &p, &f_one, C_act.data(), &m,
              d, &inc1, &f_zero, q_sub.data(), &inc1);
        return 0;
    };

    auto solve_LS = [&]() {
        memcpy(s_sub.data(), q_sub.data(), (size_t)p * sizeof(double));
        dtrsv(&uplo_U, &trans_T, &diag_N, &p, R.data(), &nld,
              s_sub.data(), &inc1);
        dtrsv(&uplo_U, &trans_N, &diag_N, &p, R.data(), &nld,
              s_sub.data(), &inc1);
    };

    auto start = chrono::high_resolution_clock::now();

    // Initial gradient w = C' * d  (x = 0)
    dgemv(&trans_T, &m, &n, &f_one, C, &m,
          d, &inc1, &f_zero, w.data(), &inc1);

    int iter = 0;
    while (true) {
        // Find max w over non-passive set
        double max_w = -numeric_limits<double>::infinity();
        for (ptrdiff_t j = 0; j < n; ++j)
            if (!P[(size_t)j] && w[(size_t)j] > max_w) max_w = w[(size_t)j];
        if (max_w <= tol) break;

        // BATCH ADD: all j not in P with w[j] > theta_add * max_w
        double t_add = max_w * theta_add;
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
            for (ptrdiff_t j = 0; j < n; ++j) s[(size_t)j] = 0.0;
            for (ptrdiff_t i = 0; i < p; ++i)
                s[(size_t)idx_P[(size_t)i]] = s_sub[(size_t)i];

            // Feasibility check
            bool has_infeasible = false;
            for (ptrdiff_t i = 0; i < p; ++i) {
                if (s_sub[(size_t)i] <= 0.0) { has_infeasible = true; break; }
            }
            if (!has_infeasible) break;

            // BATCH REMOVE
            double min_s = numeric_limits<double>::infinity();
            for (ptrdiff_t i = 0; i < p; ++i)
                if (s_sub[(size_t)i] < min_s) min_s = s_sub[(size_t)i];
            double t_rem = min_s * theta_rem;

            // Interpolation alpha
            double alpha = numeric_limits<double>::infinity();
            for (ptrdiff_t i = 0; i < p; ++i) {
                ptrdiff_t j = idx_P[(size_t)i];
                if (s[(size_t)j] < t_rem) {
                    double denom = x[(size_t)j] - s[(size_t)j];
                    if (denom > 0.0) {
                        double a = x[(size_t)j] / denom;
                        if (a < alpha) alpha = a;
                    }
                }
            }
            if (!isfinite(alpha)) alpha = 0.0;

            // x += alpha * (s - x)
            for (ptrdiff_t j = 0; j < n; ++j)
                x[(size_t)j] += alpha * (s[(size_t)j] - x[(size_t)j]);

            // Remove zero-valued passive vars where s <= 0
            ptrdiff_t p_before = p;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (P[(size_t)j] && fabs(x[(size_t)j]) < tol && s[(size_t)j] <= 0.0) {
                    P[(size_t)j] = 0;
                    x[(size_t)j] = 0.0;
                }
            }

            sync_idx_P();
            // Safety: if nothing actually got removed, break to avoid infinite loop
            if (p == p_before) break;
            if (rebuild() != 0) { iter = max_iter + 1; break; }
        }

        if (iter > max_iter) break;

        // Accept s as x
        memcpy(x.data(), s.data(), (size_t)n * sizeof(double));

        // Update gradient: r = d - C*x;  w = C^T * r
        memcpy(r.data(), d, (size_t)m * sizeof(double));
        dgemv(&trans_N, &m, &n, &f_neg_one, C, &m,
              x.data(), &inc1, &f_one, r.data(), &inc1);
        dgemv(&trans_T, &m, &n, &f_one, C, &m,
              r.data(), &inc1, &f_zero, w.data(), &inc1);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Ensure non-negativity
    for (ptrdiff_t i = 0; i < n; ++i)
        if (x[(size_t)i] < 0.0) x[(size_t)i] = 0.0;

    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    memcpy(out, x.data(), (size_t)n * sizeof(double));

    mexPrintf("NNLS FAST-NNLS (FP64, incremental Cholesky, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
