/**
 * @file nnls_gradient_projection_fp32_omp.cpp
 * @brief Non-Negative Least Squares solver using Gradient Projection method with OpenMP (FP32)
 *
 * This file implements the Gradient Projection algorithm for solving Non-Negative Least
 * Squares (NNLS) problems in single precision (FP32) with OpenMP parallelization.
 *
 * The gradient projection method features:
 *   - Simpler iterations than Active-Set
 *   - Element-wise operations ideal for parallelization
 *   - May require more iterations but each iteration is faster
 *   - FP32 provides faster computation with lower memory footprint
 *
 * The solver finds x that minimizes ||Ax - b||^2 subject to x >= 0
 *
 * References:
 *   - Vu et al. (2022). On Asymptotic Linear Convergence of Projected Gradient Descent.
 *   - Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data Parallelism: A Comparative Study. Proceedings of IEEE-MCSoC'25.
 *
 * @date 2025
 * @license MIT License
 */

#include "mex.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

/**
 * @brief Perform matrix-vector multiplication: result = A * x
 * @param A Input matrix (m x n)
 * @param x Input vector (length n)
 * @return Result vector (length m)
 */
vector<float> matVecMultiply(const vector<vector<float>>& A, const vector<float>& x) {
    size_t m = A.size();
    size_t n = A[0].size();
    vector<float> result(m, 0.0f);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        result[i] = sum;
    }
    return result;
}

/**
 * @brief Compute residual: r = Ax - b
 * @param A System matrix (m x n)
 * @param x Current solution vector (length n)
 * @param b Measurement vector (length m)
 * @return Residual vector (length m)
 */
vector<float> computeResidual(const vector<vector<float>>& A, const vector<float>& x,
                               const vector<float>& b) {
    vector<float> Ax = matVecMultiply(A, x);
    vector<float> r(Ax.size(), 0.0f);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Ax.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }
    return r;
}

/**
 * @brief Compute Euclidean norm of a vector
 * @param v Input vector
 * @return ||v||_2
 */
float vectorNorm(const vector<float>& v) {
    float norm = 0.0f;
    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < v.size(); ++i) {
        norm += v[i] * v[i];
    }
    return sqrtf(norm);
}

/**
 * @brief Solve NNLS using Gradient Projection with backtracking line search
 * @param A System matrix (m x n)
 * @param b Measurement vector (length m)
 * @param tol Convergence tolerance
 * @param max_iter Maximum number of iterations
 * @return Solution vector x (length n) with x >= 0
 */
vector<float> nnlsGradientProjection(const vector<vector<float>>& A, const vector<float>& b,
                                      float tol = 1e-6f, size_t max_iter = 2000) {
    size_t m = A.size();
    size_t n = A[0].size();

    vector<float> x(n, 0.0f);  // Initialize solution to zero
    float alpha = 0.01f;        // Initial step size
    float beta = 0.5f;          // Backtracking parameter
    float c = 0.5f;             // Armijo condition parameter

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Compute residual: r = Ax - b
        vector<float> r = computeResidual(A, x, b);

        // Compute gradient: g = A^T * r
        vector<float> g(n, 0.0f);
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < m; ++i) {
                sum += A[i][j] * r[i];
            }
            g[j] = sum;
        }

        // Compute gradient norm for convergence check
        float grad_norm = vectorNorm(g);
        if (grad_norm < tol) {
            break;
        }

        // Backtracking line search
        float step_size = alpha;
        vector<float> x_new(n);
        float f_current = vectorNorm(r) * vectorNorm(r) / 2.0f;

        for (int bt_iter = 0; bt_iter < 20; ++bt_iter) {
            // Gradient descent step with projection
            #pragma omp parallel for schedule(static)
            for (size_t j = 0; j < n; ++j) {
                x_new[j] = fmaxf(0.0f, x[j] - step_size * g[j]);
            }

            // Compute new objective value
            vector<float> r_new = computeResidual(A, x_new, b);
            float f_new = vectorNorm(r_new) * vectorNorm(r_new) / 2.0f;

            // Check Armijo condition
            float directional_deriv = 0.0f;
            #pragma omp parallel for reduction(+:directional_deriv)
            for (size_t j = 0; j < n; ++j) {
                directional_deriv += g[j] * (x_new[j] - x[j]);
            }

            if (f_new <= f_current + c * directional_deriv) {
                break;  // Sufficient decrease achieved
            }

            step_size *= beta;  // Reduce step size
        }

        // Update solution
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; ++j) {
            x[j] = x_new[j];
        }
    }

    return x;
}

/**
 * @brief MEX gateway function for MATLAB interface
 *
 * Usage: x = nnls_gradient_projection_fp32_omp(A, b, num_threads)
 *
 * Inputs:
 *   A - System matrix (m x n, single precision)
 *   b - Measurement vector (m x 1, single precision)
 *   num_threads - Number of OpenMP threads to use
 *
 * Outputs:
 *   x - Solution vector (n x 1, single precision) with non-negative entries
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Validate input arguments
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("nnlsGradientProjection:input",
                         "Three input arguments required: matrix A, vector b, and num_threads");
    }

    // Extract inputs
    const mxArray *mxA = prhs[0];
    const mxArray *mxB = prhs[1];

    size_t m = mxGetM(mxA);
    size_t n = mxGetN(mxA);

    // Convert MATLAB matrix A to C++ vector format (single precision)
    vector<vector<float>> A(m, vector<float>(n));
    float *pA = (float *)mxGetData(mxA);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = pA[i + j * m];
        }
    }

    // Convert MATLAB vector b to C++ vector format (single precision)
    vector<float> b(m);
    float *pB = (float *)mxGetData(mxB);
    for (size_t i = 0; i < m; ++i) {
        b[i] = pB[i];
    }

    // Set number of OpenMP threads
    int num_threads = static_cast<int>(mxGetScalar(prhs[2]));
    omp_set_num_threads(num_threads);

    // Record start time
    auto start = chrono::high_resolution_clock::now();

    // Solve NNLS problem
    vector<float> x = nnlsGradientProjection(A, b);

    // Record end time
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Convert result to MATLAB array (single precision)
    plhs[0] = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
    float *pOut = (float *)mxGetData(plhs[0]);
    for (size_t i = 0; i < n; ++i) {
        pOut[i] = x[i];
    }

    // Print execution time
    mexPrintf("NNLS Gradient Projection (FP32, OpenMP with %d threads) - Execution Time: %lld microseconds\n",
             num_threads, duration.count());
}
