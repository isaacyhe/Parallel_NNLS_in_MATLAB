/**
 * @file nnls_active_set_fp32_cuda.cu
 * @brief Non-Negative Least Squares solver using Active Set method with CUDA (FP32)
 *
 * This file implements an optimized Active Set algorithm for solving NNLS
 * problems on NVIDIA GPUs using CUDA with single precision (FP32).
 *
 * Key optimizations:
 *   - CUDA kernels for matrix-vector multiplication and transpose
 *   - Custom kernels for element-wise operations
 *   - Coalesced memory access patterns
 *   - CPU-based Cholesky decomposition for subproblems
 *
 * The solver finds x that minimizes ||Cx - d||^2 subject to x >= 0
 *
 * References:
 *   - Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data Parallelism.
 *
 * @date 2025
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexPrintf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(err)); \
        mexErrMsgIdAndTxt("CUDA:error", cudaGetErrorString(err)); \
    } \
} while (0)

/**
 * @brief CUDA kernel: Matrix-vector multiplication (row-major layout)
 * @param mat Input matrix (row-major, rows x cols)
 * @param vec Input vector (length cols)
 * @param result Output vector (length rows)
 * @param rows Number of rows in matrix
 * @param cols Number of columns in matrix
 */
__global__ void matVecMultiplyKernel(const float* mat, const float* vec,
                                     float* result, size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += mat[i * cols + j] * vec[j];
        }
        result[i] = sum;
    }
}

/**
 * @brief CUDA kernel: Matrix transpose (row-major to row-major)
 * @param mat Input matrix (row-major, rows x cols)
 * @param transposed Output transposed matrix (row-major, cols x rows)
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 */
__global__ void transposeKernel(const float* mat, float* transposed,
                                size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        transposed[j * rows + i] = mat[i * cols + j];
    }
}

/**
 * @brief Perform matrix-vector multiplication using CUDA
 * @param mat Input matrix (vector of rows)
 * @param vec Input vector
 * @return Result vector
 */
vector<float> matVecMultiplyCUDA(const vector<vector<float>>& mat,
                                 const vector<float>& vec) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<float> result(rows, 0.0f);

    // Allocate device memory
    float *d_mat, *d_vec, *d_result;
    CUDA_CHECK(cudaMalloc(&d_mat, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vec, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, rows * sizeof(float)));

    // Flatten matrix to row-major format
    vector<float> flat_mat(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat_mat[i * cols + j] = mat[i][j];
        }
    }

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_mat, flat_mat.data(), rows * cols * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, vec.data(), cols * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    matVecMultiplyKernel<<<gridSize, blockSize>>>(d_mat, d_vec, d_result, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, rows * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

/**
 * @brief Transpose a matrix using CUDA
 * @param mat Input matrix
 * @return Transposed matrix
 */
vector<vector<float>> transposeCUDA(const vector<vector<float>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<vector<float>> transposed(cols, vector<float>(rows));

    // Allocate device memory
    float *d_mat, *d_transposed;
    CUDA_CHECK(cudaMalloc(&d_mat, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transposed, rows * cols * sizeof(float)));

    // Flatten matrix to row-major format
    vector<float> flat_mat(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat_mat[i * cols + j] = mat[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_mat, flat_mat.data(), rows * cols * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x,
                 (cols + blockDim.y - 1) / blockDim.y);
    transposeKernel<<<gridDim, blockDim>>>(d_mat, d_transposed, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    vector<float> flat_transposed(rows * cols);
    CUDA_CHECK(cudaMemcpy(flat_transposed.data(), d_transposed,
                         rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            transposed[j][i] = flat_transposed[j * rows + i];
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_transposed));

    return transposed;
}

/**
 * @brief Solve linear system using Cholesky decomposition (CPU-based)
 * @param A Symmetric positive definite matrix
 * @param b Right-hand side vector
 * @return Solution vector x (or error indicator if decomposition fails)
 */
vector<float> choleskySolve(const vector<vector<float>>& A, const vector<float>& b) {
    size_t n = A.size();
    vector<vector<float>> L(n, vector<float>(n, 0.0f));

    // Cholesky decomposition: A = L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            float diag = A[i][i] - sum;
            if (i == j) {
                if (diag <= 0) {
                    return vector<float>(n, -1.0f);  // Error indicator
                }
                L[i][j] = sqrtf(diag);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    // Forward substitution: L * y = b
    vector<float> y(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Backward substitution: L^T * x = y
    vector<float> x(n, 0.0f);
    for (int i = n - 1; i >= 0; --i) {
        float sum = 0.0f;
        for (size_t j = i + 1; j < n; ++j) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    return x;
}

/**
 * @brief Solve NNLS using GPU-accelerated Active Set method
 * @param C System matrix (m x n)
 * @param d Measurement vector (length m)
 * @param tol Convergence tolerance
 * @return Solution vector x (length n)
 */
vector<float> nnlsActiveSetCUDA(const vector<vector<float>>& C,
                                const vector<float>& d,
                                float tol = 1e-6f) {
    size_t m = C.size();
    size_t n = C[0].size();

    vector<float> x(n, 0.0f);
    vector<bool> activeSet(n, false);
    vector<float> gradient(n, 0.0f);

    // Main Active Set loop
    while (true) {
        // Compute residual: r = d - C * x
        vector<float> r = d;
        vector<float> Cx = matVecMultiplyCUDA(C, x);
        for (size_t i = 0; i < m; ++i) {
            r[i] -= Cx[i];
        }

        // Compute gradient: gradient = C^T * r
        vector<vector<float>> Ct = transposeCUDA(C);
        gradient = matVecMultiplyCUDA(Ct, r);

        // Find the variable with maximum gradient (not in active set)
        int j_max = -1;
        float max_gradient = -numeric_limits<float>::infinity();
        for (size_t j = 0; j < n; ++j) {
            if (!activeSet[j] && gradient[j] > max_gradient) {
                max_gradient = gradient[j];
                j_max = j;
            }
        }

        // Check convergence
        if (max_gradient < tol) {
            break;
        }

        // Add variable to active set
        activeSet[j_max] = true;

        // Inner loop: solve subproblem and remove negative variables
        while (true) {
            // Extract active columns from C
            vector<vector<float>> C_active(m, vector<float>());
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    for (size_t i = 0; i < m; ++i) {
                        C_active[i].push_back(C[i][j]);
                    }
                }
            }

            // Compute normal equations: A = C_active^T * C_active
            size_t active_size = C_active[0].size();
            vector<vector<float>> A(active_size, vector<float>(active_size, 0.0f));
            for (size_t i = 0; i < active_size; ++i) {
                for (size_t j = 0; j < active_size; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < m; ++k) {
                        sum += C_active[k][i] * C_active[k][j];
                    }
                    A[i][j] = sum;
                }
            }

            // Compute right-hand side: b = C_active^T * d
            vector<float> b(active_size, 0.0f);
            for (size_t i = 0; i < active_size; ++i) {
                float sum = 0.0f;
                for (size_t k = 0; k < m; ++k) {
                    sum += C_active[k][i] * d[k];
                }
                b[i] = sum;
            }

            // Solve normal equations using Cholesky decomposition
            vector<float> x_active = choleskySolve(A, b);
            if (x_active[0] == -1.0f) {
                // Cholesky failed, return current solution
                return x;
            }

            // Map active solution back to full solution vector
            size_t idx = 0;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    x[j] = x_active[idx++];
                } else {
                    x[j] = 0.0f;
                }
            }

            // Check for negative values and remove from active set
            bool all_nonnegative = true;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j] && x[j] < 0) {
                    all_nonnegative = false;
                    activeSet[j] = false;
                }
            }

            // Exit inner loop if all values are non-negative
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
 * Usage: x = nnlsActiveSetCUDA(C, d)
 *
 * Inputs:
 *   C - System matrix (m x n)
 *   d - Measurement vector (m x 1)
 *
 * Outputs:
 *   x - Solution vector (n x 1) with non-negative entries
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Validate input arguments
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsActiveSetCUDA:input",
                         "Two input arguments required: matrix C and vector d");
    }

    // Extract inputs
    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];

    size_t m = mxGetM(mxC);
    size_t n = mxGetN(mxC);

    // Convert MATLAB matrix C to C++ vector format
    vector<vector<float>> C(m, vector<float>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = static_cast<float>(mxGetPr(mxC)[i + j * m]);
        }
    }

    // Convert MATLAB vector d to C++ vector format
    vector<float> d(m);
    for (size_t i = 0; i < m; ++i) {
        d[i] = static_cast<float>(mxGetPr(mxD)[i]);
    }

    // Record start time
    auto start = chrono::high_resolution_clock::now();

    // Solve NNLS problem on GPU
    vector<float> x = nnlsActiveSetCUDA(C, d);

    // Record end time
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Convert result to MATLAB array
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = static_cast<double>(x[i]);
    }

    // Print execution time
    mexPrintf("NNLS Active Set (FP32, CUDA) - Execution Time: %lld microseconds\n",
             duration.count());
}
