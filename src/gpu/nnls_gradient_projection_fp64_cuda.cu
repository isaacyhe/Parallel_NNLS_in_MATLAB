/**
 * @file nnls_gradient_projection_fp64_cuda.cu
 * @brief Non-Negative Least Squares solver using Gradient Projection with CUDA (FP64)
 *
 * This file implements an optimized Gradient Projection algorithm for solving NNLS
 * problems on NVIDIA GPUs using CUDA with double precision (FP64).
 *
 * Key optimizations:
 *   - CUDA kernels for matrix-vector multiplication
 *   - Custom kernels for element-wise operations
 *   - Optimized reduction for vector norm computation
 *   - Coalesced memory access patterns
 *
 * The solver finds x that minimizes ||Ax - b||^2 subject to x >= 0
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
 * @brief CUDA kernel: Matrix-vector multiplication (MATLAB column-major layout)
 * @param A System matrix (column-major, m x n)
 * @param x Input vector (length n)
 * @param result Output vector (length m)
 * @param m Number of rows
 * @param n Number of columns
 */
__global__ void matVecMultiplyKernel(const double* A, const double* x,
                                     double* result, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i + j * m] * x[j];  // Column-major indexing
        }
        result[i] = sum;
    }
}

/**
 * @brief CUDA kernel: Compute residual r = Ax - b
 * @param Ax Result of matrix-vector product (input)
 * @param b Measurement vector (input)
 * @param r Residual vector (output)
 * @param m Length of vectors
 */
__global__ void computeResidualKernel(const double* Ax, const double* b,
                                      double* r, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        r[i] = Ax[i] - b[i];
    }
}

/**
 * @brief CUDA kernel: Compute gradient g = A^T * r
 * @param A System matrix (column-major, m x n)
 * @param r Residual vector (length m)
 * @param g Gradient vector (output, length n)
 * @param m Number of rows
 * @param n Number of columns
 */
__global__ void computeGradientKernel(const double* A, const double* r,
                                      double* g, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += A[i + j * m] * r[i];  // Column-major indexing
        }
        g[j] = sum;
    }
}

/**
 * @brief CUDA kernel: Projected gradient step x = max(0, x - alpha * g)
 * @param x Current solution (input/output)
 * @param g Gradient (input)
 * @param alpha Step size
 * @param n Length of vectors
 */
__global__ void updateAndProjectKernel(double* x, const double* g,
                                       double alpha, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double new_x = x[j] - alpha * g[j];
        x[j] = (new_x < 0.0) ? 0.0 : new_x;  // Project onto non-negative orthant
    }
}

/**
 * @brief CUDA kernel: Compute squared norm of a vector (reduction step)
 * @param v Input vector
 * @param result Partial sums (one per block)
 * @param n Length of vector
 */
__global__ void vectorNormSquaredKernel(const double* v, double* result, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? v[i] * v[i] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Solve NNLS using GPU-accelerated Gradient Projection method
 * @param d_A System matrix on device (column-major, m x n)
 * @param d_b Measurement vector on device (length m)
 * @param d_x Solution vector on device (length n, initialized to zero)
 * @param m Number of rows
 * @param n Number of columns
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 */
void nnlsGradientProjectionCUDA(double* d_A, double* d_b, double* d_x,
                                int m, int n, double tol = 1e-6,
                                int max_iter = 2000) {
    // Kernel launch parameters
    int blockSize = 256;
    int numBlocksM = (m + blockSize - 1) / blockSize;
    int numBlocksN = (n + blockSize - 1) / blockSize;

    // Allocate device memory for intermediate results
    double *d_Ax, *d_r, *d_g, *d_normTemp;
    CUDA_CHECK(cudaMalloc(&d_Ax, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_g, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_normTemp, numBlocksM * sizeof(double)));

    // Solver parameters
    double alpha = 0.001;  // Step size

    // Main iteration loop
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Ax
        matVecMultiplyKernel<<<numBlocksM, blockSize>>>(d_A, d_x, d_Ax, m, n);
        CUDA_CHECK(cudaGetLastError());

        // Compute residual: r = Ax - b
        computeResidualKernel<<<numBlocksM, blockSize>>>(d_Ax, d_b, d_r, m);
        CUDA_CHECK(cudaGetLastError());

        // Compute gradient: g = A^T * r
        computeGradientKernel<<<numBlocksN, blockSize>>>(d_A, d_r, d_g, m, n);
        CUDA_CHECK(cudaGetLastError());

        // Projected gradient step: x = max(0, x - alpha * g)
        updateAndProjectKernel<<<numBlocksN, blockSize>>>(d_x, d_g, alpha, n);
        CUDA_CHECK(cudaGetLastError());

        // Check convergence using residual norm
        vectorNormSquaredKernel<<<numBlocksM, blockSize, blockSize * sizeof(double)>>>(
            d_r, d_normTemp, m);
        CUDA_CHECK(cudaGetLastError());

        // Copy partial sums to host and compute final norm
        vector<double> h_normTemp(numBlocksM);
        CUDA_CHECK(cudaMemcpy(h_normTemp.data(), d_normTemp,
                             numBlocksM * sizeof(double), cudaMemcpyDeviceToHost));

        double residualNorm = 0.0;
        for (int i = 0; i < numBlocksM; i++) {
            residualNorm += h_normTemp[i];
        }
        residualNorm = sqrt(residualNorm);

        // Check for convergence
        if (residualNorm < tol) {
            break;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_Ax));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_normTemp));
}

/**
 * @brief MEX gateway function for MATLAB interface
 *
 * Usage: x = nnlsGradientProjectionCUDA(A, b)
 *
 * Inputs:
 *   A - System matrix (m x n)
 *   b - Measurement vector (m x 1)
 *
 * Outputs:
 *   x - Solution vector (n x 1) with non-negative entries
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Validate input arguments
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsGradientProjectionCUDA:input",
                         "Two input arguments required: matrix A and vector b");
    }

    // Extract inputs
    const mxArray *mxA = prhs[0];
    const mxArray *mxB = prhs[1];

    int m = mxGetM(mxA);
    int n = mxGetN(mxA);

    // Allocate device memory
    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));

    // Copy data to device (MATLAB uses column-major layout)
    CUDA_CHECK(cudaMemcpy(d_A, mxGetPr(mxA), m * n * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, mxGetPr(mxB), m * sizeof(double),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));  // Initialize x = 0

    // Record start time
    auto start = chrono::high_resolution_clock::now();

    // Solve NNLS problem on GPU
    nnlsGradientProjectionCUDA(d_A, d_b, d_x, m, n);

    // Record end time
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Create output array and copy result from device
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    CUDA_CHECK(cudaMemcpy(mxGetPr(plhs[0]), d_x, n * sizeof(double),
                         cudaMemcpyDeviceToHost));

    // Print execution time
    mexPrintf("NNLS Gradient Projection (FP64, CUDA) - Execution Time: %lld microseconds\n",
             duration.count());

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
}
