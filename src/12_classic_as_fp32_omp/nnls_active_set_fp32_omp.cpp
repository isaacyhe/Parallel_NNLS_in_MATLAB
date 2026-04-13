/**
 * @file nnls_active_set_fp32_omp.cpp
 * @brief NNLS Active-Set solver (FP32, OpenMP + BLAS/LAPACK, incremental Cholesky)
 *
 * Single-precision variant of the incrementally-updated Cholesky Active-Set
 * solver. Each outer iteration adds one column via a triangular solve
 * (O(m*p + p^2)) instead of a full dsyrk/dpotrf (O(m*p^2 + p^3)).
 *
 * Inputs arrive as FP64 from MATLAB and are converted to FP32 internally.
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
        mexErrMsgIdAndTxt("nnls_active_set_fp32_omp:input",
            "Three inputs required: C (matrix), d (vector), num_threads");
    }

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    if (!mxIsDouble(mxC) || !mxIsDouble(mxD) || mxIsComplex(mxC) || mxIsComplex(mxD)) {
        mexErrMsgIdAndTxt("nnls_active_set_fp32_omp:input",
            "C and d must be real double precision");
    }

    ptrdiff_t m = (ptrdiff_t)mxGetM(mxC);
    ptrdiff_t n = (ptrdiff_t)mxGetN(mxC);
    if ((ptrdiff_t)mxGetM(mxD) != m || mxGetN(mxD) != 1) {
        mexErrMsgIdAndTxt("nnls_active_set_fp32_omp:input",
            "d must be a column vector of length size(C,1)");
    }

    const double* C_d = mxGetPr(mxC);
    const double* d_d = mxGetPr(mxD);

    int num_threads = (int)mxGetScalar(prhs[2]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    // Convert to FP32 (column-major preserved)
    vector<float> C((size_t)m * (size_t)n);
    vector<float> d((size_t)m);
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t k = 0; k < m * n; ++k) C[(size_t)k] = (float)C_d[(size_t)k];
    for (ptrdiff_t i = 0; i < m; ++i) d[(size_t)i] = (float)d_d[(size_t)i];

    const float tol = 1e-5f;
    const int max_outer_iter = 3 * (int)n;
    const int max_inner_iter = 3 * (int)n;

    const char uplo_U = 'U';
    const char trans_N = 'N';
    const char trans_T = 'T';
    const char diag_N = 'N';
    const float f_one = 1.0f;
    const float f_neg_one = -1.0f;
    const float f_zero = 0.0f;
    const ptrdiff_t inc1 = 1;
    const ptrdiff_t nld = n;

    vector<float> x((size_t)n, 0.0f);
    vector<unsigned char> activeSet((size_t)n, 0);
    vector<float> gradient((size_t)n, 0.0f);
    vector<float> r((size_t)m, 0.0f);

    vector<float> C_act((size_t)m * (size_t)n);
    vector<float> R((size_t)n * (size_t)n, 0.0f);
    vector<float> q_sub((size_t)n, 0.0f);
    vector<float> x_sub((size_t)n, 0.0f);
    vector<float> v_scratch((size_t)n, 0.0f);
    vector<ptrdiff_t> idx_P;
    idx_P.reserve((size_t)n);
    ptrdiff_t p = 0;

    auto rebuild = [&]() -> int {
        p = (ptrdiff_t)idx_P.size();
        if (p == 0) return 0;

        #pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < p; ++i) {
            memcpy(C_act.data() + (size_t)i * (size_t)m,
                   C.data() + (size_t)idx_P[(size_t)i] * (size_t)m,
                   (size_t)m * sizeof(float));
        }

        ssyrk(&uplo_U, &trans_T, &p, &m, &f_one,
              C_act.data(), &m, &f_zero, R.data(), &nld);

        ptrdiff_t info = 0;
        spotrf(&uplo_U, &p, R.data(), &nld, &info);
        if (info != 0) {
            float trace = 0.0f;
            for (ptrdiff_t k = 0; k < p; ++k) {
                const float* col = C_act.data() + (size_t)k * (size_t)m;
                float s = 0.0f;
                for (ptrdiff_t ii = 0; ii < m; ++ii) s += col[ii] * col[ii];
                trace += s;
            }
            float jitter = 1e-6f * (trace / (float)p + 1.0f);
            ssyrk(&uplo_U, &trans_T, &p, &m, &f_one,
                  C_act.data(), &m, &f_zero, R.data(), &nld);
            for (ptrdiff_t k = 0; k < p; ++k)
                R[(size_t)k * (size_t)nld + (size_t)k] += jitter;
            info = 0;
            spotrf(&uplo_U, &p, R.data(), &nld, &info);
        }
        if (info != 0) return (int)info;

        sgemv(&trans_T, &m, &p, &f_one, C_act.data(), &m,
              d.data(), &inc1, &f_zero, q_sub.data(), &inc1);
        return 0;
    };

    auto add_col = [&](ptrdiff_t j) -> int {
        const float* c_new = C.data() + (size_t)j * (size_t)m;

        float c_nrm2 = sdot(&m, c_new, &inc1, c_new, &inc1);
        if (c_nrm2 < 1e-20f) return -1;
        float q_new = sdot(&m, c_new, &inc1, d.data(), &inc1);

        if (p == 0) {
            memcpy(C_act.data(), c_new, (size_t)m * sizeof(float));
            R[0] = sqrtf(c_nrm2);
            q_sub[0] = q_new;
            idx_P.push_back(j);
            p = 1;
            return 0;
        }

        sgemv(&trans_T, &m, &p, &f_one, C_act.data(), &m,
              c_new, &inc1, &f_zero, v_scratch.data(), &inc1);

        strsv(&uplo_U, &trans_T, &diag_N, &p, R.data(), &nld,
              v_scratch.data(), &inc1);

        float r_nrm2 = sdot(&p, v_scratch.data(), &inc1,
                            v_scratch.data(), &inc1);
        float rho2 = c_nrm2 - r_nrm2;
        if (rho2 < 1e-6f * c_nrm2) return -1;
        float rho = sqrtf(rho2);

        memcpy(C_act.data() + (size_t)p * (size_t)m,
               c_new, (size_t)m * sizeof(float));
        for (ptrdiff_t i = 0; i < p; ++i)
            R[(size_t)p * (size_t)nld + (size_t)i] = v_scratch[(size_t)i];
        R[(size_t)p * (size_t)nld + (size_t)p] = rho;
        q_sub[(size_t)p] = q_new;

        idx_P.push_back(j);
        p++;
        return 0;
    };

    auto solve_LS = [&]() {
        memcpy(x_sub.data(), q_sub.data(), (size_t)p * sizeof(float));
        strsv(&uplo_U, &trans_T, &diag_N, &p, R.data(), &nld,
              x_sub.data(), &inc1);
        strsv(&uplo_U, &trans_N, &diag_N, &p, R.data(), &nld,
              x_sub.data(), &inc1);
    };

    auto start = chrono::high_resolution_clock::now();

    for (int outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {
        memcpy(r.data(), d.data(), (size_t)m * sizeof(float));
        sgemv(&trans_N, &m, &n, &f_neg_one, C.data(), &m,
              x.data(), &inc1, &f_one, r.data(), &inc1);

        sgemv(&trans_T, &m, &n, &f_one, C.data(), &m,
              r.data(), &inc1, &f_zero, gradient.data(), &inc1);

        ptrdiff_t j_max = -1;
        float max_grad = -numeric_limits<float>::infinity();
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (!activeSet[(size_t)j] && gradient[(size_t)j] > max_grad) {
                max_grad = gradient[(size_t)j];
                j_max = j;
            }
        }
        if (max_grad < tol) break;

        activeSet[(size_t)j_max] = 1;
        int info = add_col(j_max);
        if (info != 0) {
            activeSet[(size_t)j_max] = 0;
            break;
        }

        for (int inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {
            solve_LS();

            for (ptrdiff_t j = 0; j < n; ++j)
                if (!activeSet[(size_t)j]) x[(size_t)j] = 0.0f;
            bool all_pos = true;
            for (ptrdiff_t i = 0; i < p; ++i) {
                x[(size_t)idx_P[(size_t)i]] = x_sub[(size_t)i];
                if (x_sub[(size_t)i] < -tol) {
                    activeSet[(size_t)idx_P[(size_t)i]] = 0;
                    all_pos = false;
                }
            }
            if (all_pos) break;

            ptrdiff_t keep = 0;
            for (ptrdiff_t i = 0; i < p; ++i) {
                if (activeSet[(size_t)idx_P[(size_t)i]]) {
                    idx_P[(size_t)keep++] = idx_P[(size_t)i];
                }
            }
            idx_P.resize((size_t)keep);
            if (rebuild() != 0) break;
        }
    }

    plhs[0] = mxCreateDoubleMatrix((size_t)n, 1, mxREAL);
    double* out = mxGetPr(plhs[0]);
    for (ptrdiff_t j = 0; j < n; ++j)
        out[j] = (x[(size_t)j] > 0.0f) ? (double)x[(size_t)j] : 0.0;

    auto end_t = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_t - start);

    mexPrintf("NNLS Active-Set (FP32, incremental Cholesky, OpenMP %d threads) - Time: %lld us\n",
              num_threads, (long long)duration.count());
}
