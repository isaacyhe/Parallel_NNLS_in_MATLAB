#include "mex.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexPrintf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        mexErrMsgIdAndTxt("CUDA:Error", "CUDA runtime error"); \
    } \
}

// CUDA 核函数：矩阵向量乘法 Ax
__global__ void matVecMultiplyKernel(const double* A, const double* x, double* result, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i + j * m] * x[j]; // MATLAB 列优先
        }
        result[i] = sum;
    }
}

// CUDA 核函数：计算残差 r = Ax - b
__global__ void computeResidualKernel(const double* Ax, const double* b, double* r, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        r[i] = Ax[i] - b[i];
    }
}

// CUDA 核函数：计算梯度 g = A^T * r
__global__ void computeGradientKernel(const double* A, const double* r, double* g, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            sum += A[i + j * m] * r[i];
        }
        g[j] = sum;
    }
}

// CUDA 核函数：更新 x 并投影到非负空间
__global__ void updateAndProjectKernel(double* x, const double* g, double alpha, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        double new_x = x[j] - alpha * g[j];
        x[j] = (new_x < 0) ? 0 : new_x;
    }
}

// CUDA 核函数：计算向量范数的平方和
__global__ void vectorNormSquaredKernel(const double* v, double* result, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? v[i] * v[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) result[blockIdx.x] = sdata[0];
}

// 非负最小二乘法的 CUDA 实现
void lsqnonnegGradientProjectionCUDA(double* d_A, double* d_b, double* d_x, int m, int n, 
                                     double tol = 1e-6, int max_iter = 2000) {
    int blockSize = 256;
    double alpha = 0.001;

    double *d_Ax, *d_r, *d_g, *d_normTemp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ax, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_r, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g, n * sizeof(double)));
    int numBlocksM = (m + blockSize - 1) / blockSize;
    int numBlocksN = (n + blockSize - 1) / blockSize;
    CHECK_CUDA_ERROR(cudaMalloc(&d_normTemp, numBlocksM * sizeof(double)));

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float milliseconds = 0;

    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    for (int iter = 0; iter < max_iter; iter++) {
        // 计算 Ax
        matVecMultiplyKernel<<<numBlocksM, blockSize>>>(d_A, d_x, d_Ax, m, n);

        // 计算残差 r = Ax - b
        computeResidualKernel<<<numBlocksM, blockSize>>>(d_Ax, d_b, d_r, m);

        // 计算梯度 g = A^T * r
        computeGradientKernel<<<numBlocksN, blockSize>>>(d_A, d_r, d_g, m, n);

        // 更新 x 并投影
        updateAndProjectKernel<<<numBlocksN, blockSize>>>(d_x, d_g, alpha, n);

        // 计算残差范数
        vectorNormSquaredKernel<<<numBlocksM, blockSize, blockSize * sizeof(double)>>>(d_r, d_normTemp, m);
        std::vector<double> h_normTemp(numBlocksM);
        CHECK_CUDA_ERROR(cudaMemcpy(h_normTemp.data(), d_normTemp, numBlocksM * sizeof(double), cudaMemcpyDeviceToHost));
        double residualNorm = 0.0;
        for (int i = 0; i < numBlocksM; i++) {
            residualNorm += h_normTemp[i];
        }
        residualNorm = sqrt(residualNorm);

        if (residualNorm < tol) {
            mexPrintf("Converged after %d iterations, residual norm: %.10f\n", iter, residualNorm);
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    double microseconds = milliseconds * 1000.0;
    mexPrintf("\nExecution Time: %.2f microseconds\n", microseconds);

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_Ax));
    CHECK_CUDA_ERROR(cudaFree(d_r));
    CHECK_CUDA_ERROR(cudaFree(d_g));
    CHECK_CUDA_ERROR(cudaFree(d_normTemp));
}

// MEX 文件入口函数
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsGradientProjection:input", "Two input arguments required: matrix A and vector b.");
    }

    const mxArray *mxA = prhs[0];
    const mxArray *mxB = prhs[1];
    int m = mxGetM(mxA);
    int n = mxGetN(mxA);

    double *d_A, *d_b, *d_x;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, m * n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, mxGetPr(mxA), m * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, mxGetPr(mxB), m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_x, 0, n * sizeof(double)));

    lsqnonnegGradientProjectionCUDA(d_A, d_b, d_x, m, n);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    CHECK_CUDA_ERROR(cudaMemcpy(mxGetPr(plhs[0]), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    double* x = mxGetPr(plhs[0]);
    mexPrintf(">> disp(x);\n");
    for (int i = 0; i < n; i++) {
        if (x[i] == 0.0) {
            mexPrintf("         0\n");
        } else {
            mexPrintf("    %.4f\n", x[i]);
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_x));
}
