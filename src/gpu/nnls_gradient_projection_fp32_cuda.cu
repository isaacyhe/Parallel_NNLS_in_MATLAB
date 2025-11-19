/**
 * @file nnls_gradient_projection_fp32_cuda.cu
 * @brief Non-Negative Least Squares solver using Gradient Projection with CUDA (FP32)
 *
 * This file implements an optimized Gradient Projection algorithm for solving NNLS
 * problems on NVIDIA GPUs using CUDA with single precision (FP32).
 *
 * Key optimizations:
 *   - Persistent GPU memory allocation (minimize host-device transfers)
 *   - cuBLAS for matrix operations
 *   - Custom kernels for element-wise operations
 *   - Coalesced memory access patterns
 *
 * The solver finds x that minimizes ||Ax - b||^2 subject to x >= 0
 *
 * References:
 *   - Zhu et al. (2025). Accelerating Magnetic Particle Imaging with Data Parallelism.
 *
 * @author Parallel NNLS Team
 * @date 2025
 * @license MIT License
 */

#include "mex.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        mexPrintf("cuBLAS Error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        mexErrMsgIdAndTxt("cuBLAS:error", "cuBLAS operation failed"); \
    } \
} while (0)

/**
 * @brief CUDA kernel: Compute residual r = Ax - b (element-wise subtraction)
 * @param Ax Result of matrix-vector product (input)
 * @param b Measurement vector (input)
 * @param r Residual vector (output)
 * @param m Length of vectors
 */
__global__ void computeResidualKernel(const float* Ax, const float* b, float* r, size_t m) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        r[i] = Ax[i] - b[i];
    }
}

/**
 * @brief CUDA kernel: Projected gradient step x_new = max(0, x - alpha * g)
 * @param x Current solution (input)
 * @param g Gradient (input)
 * @param x_new Updated solution (output)
 * @param alpha Step size
 * @param n Length of vectors
 */
__global__ void projectedGradientStepKernel(const float* x, const float* g,
                                            float* x_new, float alpha, size_t n) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float val = x[j] - alpha * g[j];
        x_new[j] = fmaxf(0.0f, val);  // Project onto non-negative orthant
    }
}

/**
 * @brief CUDA kernel: Compute dot product for convergence check
 * @param g Gradient vector (input)
 * @param dx Step direction (x_new - x) (input)
 * @param result Partial sums (output)
 * @param n Length of vectors
 */
__global__ void dotProductKernel(const float* g, const float* dx, float* result, size_t n) {
    __shared__ float sdata[256];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g[i] * dx[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

/**
 * @brief Solve NNLS using GPU-accelerated Gradient Projection
 * @param A System matrix (m x n, row-major on host)
 * @param b Measurement vector (length m)
 * @param m Number of rows
 * @param n Number of columns
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 * @return Solution vector x (length n)
 */
vector<float> nnlsGradientProjectionCUDA(const vector<vector<float>>& A,
                                         const vector<float>& b,
                                         size_t m, size_t n,
                                         float tol = 1e-6f,
                                         size_t max_iter = 2000) {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Flatten matrix A to column-major format for cuBLAS
    vector<float> A_flat(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_flat[j * m + i] = A[i][j];  // Column-major
        }
    }

    // Allocate device memory
    float *d_A, *d_b, *d_x, *d_x_new, *d_Ax, *d_r, *d_g, *d_dot;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_new, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Ax, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));  // Initialize x = 0

    // cuBLAS constants
    const float alpha_blas = 1.0f;
    const float beta_blas = 0.0f;
    const float neg_one = -1.0f;

    // Kernel launch parameters
    int blockSize = 256;
    int gridSize_m = (m + blockSize - 1) / blockSize;
    int gridSize_n = (n + blockSize - 1) / blockSize;

    // Solver parameters
    float step_size = 0.01f;
    float backtrack_beta = 0.5f;

    // Main iteration loop
    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Compute Ax using cuBLAS: d_Ax = d_A * d_x
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, m, n,
                                &alpha_blas, d_A, m, d_x, 1,
                                &beta_blas, d_Ax, 1));

        // Compute residual: d_r = d_Ax - d_b
        computeResidualKernel<<<gridSize_m, blockSize>>>(d_Ax, d_b, d_r, m);
        CUDA_CHECK(cudaGetLastError());

        // Compute gradient: d_g = A^T * d_r
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, m, n,
                                &alpha_blas, d_A, m, d_r, 1,
                                &beta_blas, d_g, 1));

        // Check gradient norm for convergence
        float grad_norm;
        CUBLAS_CHECK(cublasSnrm2(handle, n, d_g, 1, &grad_norm));

        if (grad_norm < tol) {
            break;
        }

        // Projected gradient step: x_new = max(0, x - step_size * g)
        projectedGradientStepKernel<<<gridSize_n, blockSize>>>(
            d_x, d_g, d_x_new, step_size, n);
        CUDA_CHECK(cudaGetLastError());

        // Update solution: x = x_new
        CUDA_CHECK(cudaMemcpy(d_x, d_x_new, n * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Copy result back to host
    vector<float> x(n);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_new));
    CUDA_CHECK(cudaFree(d_Ax));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_dot));
    CUBLAS_CHECK(cublasDestroy(handle));

    return x;
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

    size_t m = mxGetM(mxA);
    size_t n = mxGetN(mxA);

    // Convert MATLAB matrix A to C++ vector format
    vector<vector<float>> A(m, vector<float>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = static_cast<float>(mxGetPr(mxA)[i + j * m]);
        }
    }

    // Convert MATLAB vector b to C++ vector format
    vector<float> b(m);
    for (size_t i = 0; i < m; ++i) {
        b[i] = static_cast<float>(mxGetPr(mxB)[i]);
    }

    // Record start time
    auto start = chrono::high_resolution_clock::now();

    // Solve NNLS problem on GPU
    vector<float> x = nnlsGradientProjectionCUDA(A, b, m, n);

    // Record end time
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Convert result to MATLAB array
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = static_cast<double>(x[i]);
    }

    // Print execution time
    mexPrintf("NNLS Gradient Projection (FP32, CUDA) - Execution Time: %lld microseconds\n",
             duration.count());
}
