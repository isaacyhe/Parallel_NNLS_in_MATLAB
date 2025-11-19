/**
 * @file nnls_active_set_fp64_omp.cpp
 * @brief Non-Negative Least Squares solver using Active-Set method with OpenMP (FP64)
 *
 * This file implements the Active-Set algorithm for solving Non-Negative Least Squares
 * (NNLS) problems in double precision (FP64) with OpenMP parallelization for multi-core CPUs.
 *
 * The solver finds x that minimizes ||Cx - d||^2 subject to x >= 0
 *
 * References:
 *   - Lawson, C. L., & Hanson, R. J. (1995). Solving Least Squares Problems. SIAM.
 *   - Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data Parallelism.
 *
 * @author Parallel NNLS Team
 * @date 2025
 * @license MIT License
 */

#include "mex.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>

using namespace std;

/**
 * @brief Perform matrix-vector multiplication (parallel)
 * @param mat Input matrix (rows x cols)
 * @param vec Input vector (length cols)
 * @return Result vector (length rows)
 */
vector<double> matVecMultiply(const vector<vector<double>>& mat, const vector<double>& vec) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<double> result(rows, 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += mat[i][j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

/**
 * @brief Transpose a matrix
 * @param mat Input matrix (rows x cols)
 * @return Transposed matrix (cols x rows)
 */
vector<vector<double>> transpose(const vector<vector<double>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<vector<double>> transposed(cols, vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

/**
 * @brief Solve a symmetric positive definite system using Cholesky decomposition
 * @param A Coefficient matrix (symmetric positive definite)
 * @param b Right-hand side vector
 * @return Solution vector x
 */
vector<double> choleskySolve(const vector<vector<double>>& A, const vector<double>& b) {
    size_t n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0.0));

    // Cholesky decomposition: A = L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                double diag = A[i][i] - sum;
                if (diag <= 0.0) {
                    mexWarnMsgTxt("Matrix is not positive definite in Cholesky decomposition");
                    L[i][j] = 1e-10; // Regularize to avoid numerical issues
                } else {
                    L[i][j] = sqrt(diag);
                }
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    // Forward substitution: L * y = b
    vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Backward substitution: L^T * x = y
    vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    return x;
}

/**
 * @brief Solve Non-Negative Least Squares using Active-Set method
 * @param C System matrix (m x n)
 * @param d Measurement vector (length m)
 * @param tol Convergence tolerance
 * @return Solution vector x (length n) with x >= 0
 */
vector<double> nnlsActiveSet(const vector<vector<double>>& C, const vector<double>& d, double tol = 1e-6) {
    size_t m = C.size();
    size_t n = C[0].size();

    // Initialize variables
    vector<double> x(n, 0.0);           // Solution vector, initialized to zero
    vector<bool> activeSet(n, false);   // Active set indicator
    vector<double> gradient(n, 0.0);    // Gradient vector

    // Main loop: iterate until optimality conditions are met
    while (true) {
        // Compute residual: r = d - C*x
        vector<double> r = d;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += C[i][j] * x[j];
            }
            r[i] -= sum;
        }

        // Compute gradient: g = C^T * r
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < m; ++i) {
                sum += C[i][j] * r[i];
            }
            gradient[j] = sum;
        }

        // Find the variable with maximum gradient among inactive variables
        int j_max = -1;
        double max_gradient = -numeric_limits<double>::infinity();
        for (size_t j = 0; j < n; ++j) {
            if (!activeSet[j] && gradient[j] > max_gradient) {
                max_gradient = gradient[j];
                j_max = j;
            }
        }

        // Check for convergence
        if (max_gradient < tol) {
            break;
        }

        // Add variable to active set
        activeSet[j_max] = true;

        // Inner loop: solve with current active set
        while (true) {
            // Extract active columns from C
            vector<vector<double>> C_active(m, vector<double>());
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    for (size_t i = 0; i < m; ++i) {
                        C_active[i].push_back(C[i][j]);
                    }
                }
            }

            // Compute C_active^T * C_active
            size_t active_size = C_active[0].size();
            vector<vector<double>> A(active_size, vector<double>(active_size, 0.0));
            #pragma omp parallel for schedule(static) collapse(2)
            for (size_t i = 0; i < active_size; ++i) {
                for (size_t j = 0; j < active_size; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < m; ++k) {
                        sum += C_active[k][i] * C_active[k][j];
                    }
                    A[i][j] = sum;
                }
            }

            // Compute C_active^T * d
            vector<double> b(active_size, 0.0);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < active_size; ++i) {
                double sum = 0.0;
                for (size_t k = 0; k < m; ++k) {
                    sum += C_active[k][i] * d[k];
                }
                b[i] = sum;
            }

            // Solve reduced system: A * x_active = b
            vector<double> x_active = choleskySolve(A, b);

            // Update full solution vector
            size_t idx = 0;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    x[j] = x_active[idx++];
                } else {
                    x[j] = 0.0;
                }
            }

            // Check non-negativity constraints
            bool all_nonnegative = true;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j] && x[j] < -tol) {  // Use tolerance for numerical stability
                    all_nonnegative = false;
                    activeSet[j] = false;  // Remove from active set
                }
            }

            if (all_nonnegative) {
                break;
            }
        }
    }

    return x;
}

/**
 * @brief MEX gateway function for MATLAB interface
 *
 * Usage: x = nnlsActiveSet(C, d, num_threads)
 *
 * Inputs:
 *   C - System matrix (m x n)
 *   d - Measurement vector (m x 1)
 *   num_threads - Number of OpenMP threads to use
 *
 * Outputs:
 *   x - Solution vector (n x 1) with non-negative entries
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Validate input arguments
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("nnlsActiveSet:input",
                         "Three input arguments required: matrix C, vector d, and num_threads");
    }

    // Extract inputs
    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];

    size_t m = mxGetM(mxC);
    size_t n = mxGetN(mxC);

    // Convert MATLAB matrix C to C++ vector format
    vector<vector<double>> C(m, vector<double>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = mxGetPr(mxC)[i + j * m];
        }
    }

    // Convert MATLAB vector d to C++ vector format
    vector<double> d(m);
    for (size_t i = 0; i < m; ++i) {
        d[i] = mxGetPr(mxD)[i];
    }

    // Set number of OpenMP threads
    int num_threads = static_cast<int>(mxGetScalar(prhs[2]));
    omp_set_num_threads(num_threads);

    // Record start time
    auto start = chrono::high_resolution_clock::now();

    // Solve NNLS problem
    vector<double> x = nnlsActiveSet(C, d);

    // Record end time
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Convert result to MATLAB array
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = x[i];
    }

    // Print execution time
    mexPrintf("NNLS Active-Set (FP64, OpenMP with %d threads) - Execution Time: %lld microseconds\n",
             num_threads, duration.count());
}
