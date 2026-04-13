/**
 * @file nnls_active_set_fp64_omp.cpp
 * @brief NNLS Active-Set solver (FP64, OpenMP + BLAS/LAPACK, incremental Cholesky)
 *
 * Classic Lawson-Hanson Active-Set method with an *incrementally updated*
 * Cholesky factor of the active-set Gram matrix:
 *     R^T R = C_act^T * C_act
 *
 * Each outer iteration adds one column. Instead of recomputing R from
 * scratch with dsyrk + dpotrf (O(m*p^2 + p^3)), we extend R with a single
 * triangular solve (O(m*p + p^2)):
 *
 *     v     = C_act^T * c_new         (dgemv, O(m*p))
 *     r_new = R^-T v                   (dtrsv, O(p^2))
 *     rho   = sqrt(||c_new||^2 - ||r_new||^2)
 *     R'    = [[R, r_new], [0, rho]]
 *
 * The inner feasibility loop only requires two triangular solves per LS solve
 * (forward R^T y = q, back R x = y). A full rebuild (dsyrk + dpotrf) is only
 * triggered when a feasibility step removes one or more columns.
 *
 * References:
 *   - Lawson, C. L., & Hanson, R. J. (1995). Solving Least Squares Problems.
 *   - Golub & Van Loan, Matrix Computations, 4e, §6.5 (Cholesky updates).
 *   - Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data
 *     Parallelism: A Comparative Study. Proceedings of IEEE-MCSoC'25.
 *
 * @date 2025
 * @license MIT License
 */

#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("nnls_active_set_fp64_omp:input",
            "Three inputs required: C (matrix), d (vector), num_threads");
    }

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD)) {
        mexErrMsgIdAndTxt("nnls_active_set_fp64_omp:input",
            "C and d must be real double precision");
    }

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1) {
        mexErrMsgIdAndTxt("nnls_active_set_fp64_omp:input",
            "d must be a column vector of length size(C,1)");
    }

    const double* C = mxGetPr(mxC);   // column-major (m x n)
    const double* d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    const double tol = 1e-6;
    const int max_outer_iter = 3 * (int)n;
    const int max_inner_iter = 3 * (int)n;

    // BLAS/LAPACK constants
    const char uplo_U = 'U';
    const char trans_N = 'N';
    const char trans_T = 'T';
    const char diag_N = 'N';
    const double f_one = 1.0;
    const double f_neg_one = -1.0;
    const double f_zero = 0.0;
    const ptrdiff_t inc1 = 1;
    const ptrdiff_t nld = n;   // leading dim of R buffer

    // Solver state
    vector<double> x((size_t)n, 0.0);
    vector<unsigned char> activeSet((size_t)n, 0);
    vector<double> gradient((size_t)n, 0.0);
    vector<double> r((size_t)m, 0.0);

    // Active-set / Cholesky state
    vector<double> C_act((size_t)m * (size_t)n);   // m x p, grows as p grows
    vector<double> R((size_t)n * (size_t)n, 0.0);  // upper tri, leading dim n
    vector<double> q_sub((size_t)n, 0.0);          // C_act^T * d
    vector<double> x_sub((size_t)n, 0.0);          // sub-problem solution
    vector<double> v_scratch((size_t)n, 0.0);      // scratch for add-column
    vector<ptrdiff_t> idx_P;
    idx_P.reserve((size_t)n);
    ptrdiff_t p = 0;

    // --- Full rebuild of R, q_sub, C_act from current idx_P ---
    auto rebuild = [&]() -> int {
        p = (ptrdiff_t)idx_P.size();
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
            // Add a small ridge and retry
            double trace = 0.0;
            for (ptrdiff_t k = 0; k < p; ++k) {
                const double* col = C_act.data() + (size_t)k * (size_t)m;
                double s = 0.0;
                for (ptrdiff_t ii = 0; ii < m; ++ii) s += col[ii] * col[ii];
                trace += s;
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

    // --- Try to append column j to the active set via incremental Cholesky ---
    // Returns 0 on success, nonzero if rank-deficient (caller should rebuild or skip).
    auto add_col = [&](ptrdiff_t j) -> int {
        const double* c_new = C + (size_t)j * (size_t)m;

        double c_nrm2 = ddot(&m, c_new, &inc1, c_new, &inc1);
        if (c_nrm2 < 1e-30) return -1;
        double q_new = ddot(&m, c_new, &inc1, d, &inc1);

        if (p == 0) {
            memcpy(C_act.data(), c_new, (size_t)m * sizeof(double));
            R[0] = sqrt(c_nrm2);
            q_sub[0] = q_new;
            idx_P.push_back(j);
            p = 1;
            return 0;
        }

        // v = C_act[:, :p]^T * c_new
        dgemv(&trans_T, &m, &p, &f_one, C_act.data(), &m,
              c_new, &inc1, &f_zero, v_scratch.data(), &inc1);

        // Solve R^T r_new = v   (in-place on v_scratch)
        dtrsv(&uplo_U, &trans_T, &diag_N, &p, R.data(), &nld,
              v_scratch.data(), &inc1);

        double r_nrm2 = ddot(&p, v_scratch.data(), &inc1,
                             v_scratch.data(), &inc1);
        double rho2 = c_nrm2 - r_nrm2;
        if (rho2 < 1e-12 * c_nrm2) return -1;   // rank-deficient
        double rho = sqrt(rho2);

        // Append c_new as column p of C_act
        memcpy(C_act.data() + (size_t)p * (size_t)m,
               c_new, (size_t)m * sizeof(double));
        // Append r_new and rho to R
        for (ptrdiff_t i = 0; i < p; ++i)
            R[(size_t)p * (size_t)nld + (size_t)i] = v_scratch[(size_t)i];
        R[(size_t)p * (size_t)nld + (size_t)p] = rho;
        // Append q_new to q_sub
        q_sub[(size_t)p] = q_new;

        idx_P.push_back(j);
        p++;
        return 0;
    };

    // --- Solve R^T R x_sub = q_sub ---
    auto solve_LS = [&]() {
        memcpy(x_sub.data(), q_sub.data(), (size_t)p * sizeof(double));
        dtrsv(&uplo_U, &trans_T, &diag_N, &p, R.data(), &nld,
              x_sub.data(), &inc1);   // forward
        dtrsv(&uplo_U, &trans_N, &diag_N, &p, R.data(), &nld,
              x_sub.data(), &inc1);   // back
    };

    auto start = chrono::high_resolution_clock::now();

    for (int outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {
        // r = d - C*x
        memcpy(r.data(), d, (size_t)m * sizeof(double));
        dgemv(&trans_N, &m, &n, &f_neg_one, C, &m,
              x.data(), &inc1, &f_one, r.data(), &inc1);

        // gradient = C^T * r
        dgemv(&trans_T, &m, &n, &f_one, C, &m,
              r.data(), &inc1, &f_zero, gradient.data(), &inc1);

        // Select j_max: most-positive gradient among inactive
        ptrdiff_t j_max = -1;
        double max_grad = -numeric_limits<double>::infinity();
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (!activeSet[(size_t)j] && gradient[(size_t)j] > max_grad) {
                max_grad = gradient[(size_t)j];
                j_max = j;
            }
        }
        if (max_grad < tol) break;

        // Append j_max incrementally
        activeSet[(size_t)j_max] = 1;
        int info = add_col(j_max);
        if (info != 0) {
            // Rank-deficient — back out and stop (rare)
            activeSet[(size_t)j_max] = 0;
            break;
        }

        // Inner feasibility loop
        for (int inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {
            solve_LS();

            // Zero inactive entries, fill active ones, check non-negativity
            for (ptrdiff_t j = 0; j < n; ++j)
                if (!activeSet[(size_t)j]) x[(size_t)j] = 0.0;
            bool all_pos = true;
            for (ptrdiff_t i = 0; i < p; ++i) {
                x[(size_t)idx_P[(size_t)i]] = x_sub[(size_t)i];
                if (x_sub[(size_t)i] < -tol) {
                    activeSet[(size_t)idx_P[(size_t)i]] = 0;
                    all_pos = false;
                }
            }
            if (all_pos) break;

            // Shrink idx_P to surviving active indices, then rebuild
            ptrdiff_t keep = 0;
            for (ptrdiff_t i = 0; i < p; ++i) {
                if (activeSet[(size_t)idx_P[(size_t)i]]) {
                    idx_P[(size_t)keep++] = idx_P[(size_t)i];
                }
            }
            idx_P.resize((size_t)keep);
            if (rebuild() != 0) break;   // severe rank deficiency
        }
    }

    // Output (clip tiny negatives from roundoff)
    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (ptrdiff_t j = 0; j < n; ++j)
        out[j] = (x[(size_t)j] > 0.0) ? x[(size_t)j] : 0.0;

    auto end_t = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_t - start);

    mexPrintf("NNLS Active-Set (FP64, incremental Cholesky, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
