#include "nnls.h"

// Utility Function for Error Handling
inline cudaError_t checkCuda(cudaError_t result, const char *fn) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error in " << fn << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

// Kernel Update to CUDA 12 - Active-Set Method (Compatible with MATLAB lsqnonneg)
__global__ void solveNNLS_ActiveSet(float *d_A, float *d_At, float *d_x, float *d_b,
                                    float *d_R, int *nIters, int *lsIters) {
    int systemIdx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (systemIdx >= NSYS) return;

    // Solve for each system of Ax=b in parallel
    float *A = &d_A[systemIdx * MATRIX_DIM * MATRIX_DIM];
    float *x = &d_x[systemIdx * MATRIX_DIM];
    float *b = &d_b[systemIdx * MATRIX_DIM];
    float *R = &d_R[systemIdx * MATRIX_DIM];

    // Initialize x to zero
    for (int j = tid; j < MATRIX_DIM; j += stride) {
        x[j] = 0.0f;
    }
    __syncthreads();

    // Active-Set Iterative Process (MATLAB Compatible)
    bool isOptimal = false;
    for (int iter = 0; iter < MAX_ITER_NNLS && !isOptimal; ++iter) {
        // Compute A' * (b - A * x)
        for (int j = tid; j < MATRIX_DIM; j += stride) {
            float r = 0.0f;
            for (int k = 0; k < MATRIX_DIM; ++k) {
                r += A[k * MATRIX_DIM + j] * (b[k] - A[k * MATRIX_DIM + j] * x[j]);
            }
            R[j] = r;
        }
        __syncthreads();

        // Update x based on active set
        isOptimal = true;
        for (int j = tid; j < MATRIX_DIM; j += stride) {
            if (R[j] > 0) {
                x[j] += R[j];
                isOptimal = false;
            }
            x[j] = max(0.0f, x[j]);
        }
        __syncthreads();
    }
}

// MATLAB-like Interface Function for lsqnonneg and Kernel Launcher
void NNLS(float *h_C, float *h_d, float *h_x, int m, int n) {
    float *d_C, *d_d, *d_x;
    size_t matrixSize = m * n * sizeof(float);
    size_t vectorSize = n * sizeof(float);

    // Allocate Device Memory
    checkCuda(cudaMalloc(&d_C, matrixSize), "cudaMalloc - d_C");
    checkCuda(cudaMalloc(&d_d, vectorSize), "cudaMalloc - d_d");
    checkCuda(cudaMalloc(&d_x, vectorSize), "cudaMalloc - d_x");
      

    // Copy Host to Device
    checkCuda(cudaMemcpy(d_C, h_C, matrixSize, cudaMemcpyHostToDevice), "cudaMemcpy - h_C to d_C");
    checkCuda(cudaMemcpy(d_d, h_d, vectorSize, cudaMemcpyHostToDevice), "cudaMemcpy - h_d to d_d");

    // Launch Kernels
    float *d_R;
    int *nIters, *lsIters;
    checkCuda(cudaMalloc(&d_R, vectorSize), "cudaMalloc - d_R");
    checkCuda(cudaMalloc(&nIters, sizeof(int)), "cudaMalloc - nIters");
    checkCuda(cudaMalloc(&lsIters, sizeof(int)), "cudaMalloc - lsIters");

    // Active-Set Method
    int blockSize = 256; // Reduced number of threads per block to avoid out-of-bound issues
    int numBlocks = NSYS; // Number of blocks to launch

    solveNNLS_ActiveSet<<<numBlocks, blockSize>>>(d_C, nullptr, d_x, d_d, d_R, nIters, lsIters);
    checkCuda(cudaGetLastError(), "Kernel Launch - Active-Set");
    checkCuda(cudaDeviceSynchronize(), "Kernel Sync - Active-Set");

    // Copy Result Back to Host
    checkCuda(cudaMemcpy(h_x, d_x, vectorSize, cudaMemcpyDeviceToHost), "cudaMemcpy - d_x to h_x");

    // Free Device Memory
    cudaFree(d_C);
    cudaFree(d_d);
    cudaFree(d_x);
    cudaFree(d_R);
    cudaFree(nIters);
    cudaFree(lsIters);
}

