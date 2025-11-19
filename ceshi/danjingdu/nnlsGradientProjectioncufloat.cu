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
__global__ void matVecMultiplyKernel(const float* A, const float* x, float* result, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i + j * m] * x[j]; // MATLAB 列优先
        }
        result[i] = sum;
    }
}

// CUDA 核函数：计算残差 r = Ax - b
__global__ void computeResidualKernel(const float* Ax, const float* b, float* r, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        r[i] = Ax[i] - b[i];
    }
}

// CUDA 核函数：计算梯度 g = A^T * r
__global__ void computeGradientKernel(const float* A, const float* r, float* g, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += A[i + j * m] * r[i];
        }
        g[j] = sum;
    }
}

// CUDA 核函数：更新 x 并投影到非负空间
__global__ void updateAndProjectKernel(float* x, const float* g, float alpha, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float new_x = x[j] - alpha * g[j];
        x[j] = (new_x < 0) ? 0 : new_x;
    }
}

// CUDA 核函数：计算向量范数的平方和
__global__ void vectorNormSquaredKernel(const float* v, float* result, int n) {
    extern __shared__ float sdata[];
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
void lsqnonnegGradientProjectionCUDA(float* d_A, float* d_b, float* d_x, int m, int n, 
                                     float tol = 1e-6f, int max_iter = 2000) {
    int blockSize = 256;
    float alpha = 0.001f;

    float *d_Ax, *d_r, *d_g, *d_normTemp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ax, m * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_r, m * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_g, n * sizeof(float)));
    int numBlocksM = (m + blockSize - 1) / blockSize;
    int numBlocksN = (n + blockSize - 1) / blockSize;
    CHECK_CUDA_ERROR(cudaMalloc(&d_normTemp, numBlocksM * sizeof(float)));

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
        vectorNormSquaredKernel<<<numBlocksM, blockSize, blockSize * sizeof(float)>>>(d_r, d_normTemp, m);
        std::vector<float> h_normTemp(numBlocksM);
        CHECK_CUDA_ERROR(cudaMemcpy(h_normTemp.data(), d_normTemp, numBlocksM * sizeof(float), cudaMemcpyDeviceToHost));
        float residualNorm = 0.0f;
        for (int i = 0; i < numBlocksM; i++) {
            residualNorm += h_normTemp[i];
        }
        residualNorm = sqrtf(residualNorm);

        if (residualNorm < tol) {
            mexPrintf("Converged after %d iterations, residual norm: %.10f\n", iter, residualNorm);
            break;
        }
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    float microseconds = milliseconds * 1000.0f;
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

    float *d_A, *d_b, *d_x;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, m * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, m * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n * sizeof(float)));

    // MATLAB 输入是 double 类型，需要转换为 float
    std::vector<float> h_A(m * n);
    std::vector<float> h_b(m);
    double* A_ptr = mxGetPr(mxA);
    double* b_ptr = mxGetPr(mxB);
    for (int i = 0; i < m * n; i++) h_A[i] = static_cast<float>(A_ptr[i]);
    for (int i = 0; i < m; i++) h_b[i] = static_cast<float>(b_ptr[i]);

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), m * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_x, 0, n * sizeof(float)));

    lsqnonnegGradientProjectionCUDA(d_A, d_b, d_x, m, n);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    std::vector<float> h_x(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    double* out_ptr = mxGetPr(plhs[0]);
    for (int i = 0; i < n; i++) out_ptr[i] = static_cast<double>(h_x[i]);

    mexPrintf(">> disp(x);\n");
    for (int i = 0; i < n; i++) {
        if (h_x[i] == 0.0f) {
            mexPrintf("         0\n");
        } else {
            mexPrintf("    %.4f\n", h_x[i]);
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_x));
}
