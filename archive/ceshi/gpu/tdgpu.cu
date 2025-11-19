#include "mex.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexPrintf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        mexErrMsgIdAndTxt("CUDA:error", cudaGetErrorString(err)); \
    } \
} while (0)

// GPU 数据管理结构体
struct GPUData {
    double *d_C, *d_Ct, *d_d, *d_x, *d_r, *d_gradient, *d_C_active, *d_A, *d_b, *d_x_active;
    bool *d_activeSet;
    int *d_j_max;
    double *d_max_gradient;
    size_t m, n, max_active_size;
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    int *d_info;
    double *d_work;
    size_t lwork;

    GPUData(const std::vector<std::vector<double>>& C, const std::vector<double>& d)
        : m(C.size()), n(C[0].size()), max_active_size(n) {
        // 分配 GPU 内存
        CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ct, n * m * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_d, m * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_r, m * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gradient, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_C_active, m * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x_active, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_activeSet, n * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_j_max, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_max_gradient, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // 初始化 C 和 d
        std::vector<double> flat_C(m * n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                flat_C[i * n + j] = C[i][j];
        CUDA_CHECK(cudaMemcpy(d_C, flat_C.data(), m * n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_d, d.data(), m * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_activeSet, 0, n * sizeof(bool)));

        // 初始化 cuBLAS 和 cuSOLVER
        cublasCreate(&cublasHandle);
        cusolverDnCreate(&cusolverHandle);

        // 分配 cuSOLVER 工作空间
        cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER, n, d_A, n, &lwork);
        CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
    }

    ~GPUData() {
        cublasDestroy(cublasHandle);
        cusolverDnDestroy(cusolverHandle);
        cudaFree(d_C); cudaFree(d_Ct); cudaFree(d_d); cudaFree(d_x); cudaFree(d_r);
        cudaFree(d_gradient); cudaFree(d_C_active); cudaFree(d_A); cudaFree(d_b);
        cudaFree(d_x_active); cudaFree(d_activeSet); cudaFree(d_j_max);
        cudaFree(d_max_gradient); cudaFree(d_info); cudaFree(d_work);
    }
};

// 矩阵转置内核（优化版，使用共享内存）
__global__ void transposeKernel(const double* mat, double* transposed, size_t rows, size_t cols) {
    __shared__ double tile[32][33]; // +1 避免银行冲突
    size_t x = blockIdx.x * 32 + threadIdx.x;
    size_t y = blockIdx.y * 32 + threadIdx.y;

    if (x < rows && y < cols) {
        tile[threadIdx.y][threadIdx.x] = mat[x * cols + y];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < cols && y < rows) {
        transposed[x * rows + y] = tile[threadIdx.x][threadIdx.y];
    }
}

// 构造 C_active 矩阵的内核
__global__ void buildCActiveKernel(const double* d_C, const bool* d_activeSet, double* d_C_active,
                                   size_t m, size_t n, size_t active_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = 0;
    if (i < m) {
        for (size_t j = 0; j < n; ++j) {
            if (d_activeSet[j]) {
                d_C_active[i * active_size + idx] = d_C[i * n + j];
                idx++;
            }
        }
    }
}

// 寻找最大梯度的内核（简化版，需进一步优化为归约）
__global__ void findMaxGradientKernel(const double* d_gradient, const bool* d_activeSet, size_t n,
                                     int* d_j_max, double* d_max_gradient) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && !d_activeSet[j]) {
        double grad = d_gradient[j];
        if (grad > *d_max_gradient) {
            atomicMax((int*)d_j_max, (int)j);
            atomicExch(d_max_gradient, grad);
        }
    }
}

std::vector<double> nnlsActiveSetCUDA(const std::vector<std::vector<double>>& C,
                                      const std::vector<double>& d, double tol = 1e-6) {
    GPUData gpu_data(C, d);
    size_t m = gpu_data.m, n = gpu_data.n;
    std::vector<double> x(n, 0.0);
    double alpha, beta;

    // 计算 C^T 一次并缓存
    dim3 blockDim(32, 32);
    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    transposeKernel<<<gridDim, blockDim>>>(gpu_data.d_C, gpu_data.d_Ct, m, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    while (true) {
        // 计算 r = d - C * x
        alpha = -1.0; beta = 1.0;
        CUDA_CHECK(cublasDcopy(gpu_data.cublasHandle, m, gpu_data.d_d, 1, gpu_data.d_r, 1));
        cublasDgemv(gpu_data.cublasHandle, CUBLAS_OP_N, m, n, &alpha, gpu_data.d_C, m,
                    gpu_data.d_x, 1, &beta, gpu_data.d_r, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计算 gradient = C^T * r
        alpha = 1.0; beta = 0.0;
        cublasDgemv(gpu_data.cublasHandle, CUBLAS_OP_N, n, m, &alpha, gpu_data.d_Ct, n,
                    gpu_data.d_r, 1, &beta, gpu_data.d_gradient, 1);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 寻找最大梯度
        CUDA_CHECK(cudaMemset(gpu_data.d_j_max, -1, sizeof(int)));
        CUDA_CHECK(cudaMemset(gpu_data.d_max_gradient, 0, sizeof(double)));
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        findMaxGradientKernel<<<gridSize, blockSize>>>(gpu_data.d_gradient, gpu_data.d_activeSet,
                                                      n, gpu_data.d_j_max, gpu_data.d_max_gradient);
        CUDA_CHECK(cudaDeviceSynchronize());

        int j_max;
        double max_gradient;
        CUDA_CHECK(cudaMemcpy(&j_max, gpu_data.d_j_max, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&max_gradient, gpu_data.d_max_gradient, sizeof(double), cudaMemcpyDeviceToHost));

        if (max_gradient < tol || j_max == -1) {
            break;
        }

        // 更新活动集
        bool true_val = true;
        CUDA_CHECK(cudaMemcpy(&gpu_data.d_activeSet[j_max], &true_val, sizeof(bool), cudaMemcpyHostToDevice));

        while (true) {
            // 计算活动集大小
            std::vector<bool> activeSet(n);
            CUDA_CHECK(cudaMemcpy(activeSet.data(), gpu_data.d_activeSet, n * sizeof(bool), cudaMemcpyDeviceToHost));
            size_t active_size = std::count(activeSet.begin(), activeSet.end(), true);

            if (active_size == 0) break;

            // 构造 C_active
            buildCActiveKernel<<<(m + 255) / 256, 256>>>(gpu_data.d_C, gpu_data.d_activeSet,
                                                        gpu_data.d_C_active, m, n, active_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 计算 A = C_active^T * C_active
            alpha = 1.0; beta = 0.0;
            cublasDgemm(gpu_data.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, active_size, active_size, m,
                        &alpha, gpu_data.d_C_active, m, gpu_data.d_C_active, m, &beta, gpu_data.d_A, active_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 计算 b = C_active^T * d
            cublasDgemv(gpu_data.cublasHandle, CUBLAS_OP_T, m, active_size, &alpha, gpu_data.d_C_active, m,
                        gpu_data.d_d, 1, &beta, gpu_data.d_b, 1);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Cholesky 分解和求解
            cusolverDnDpotrf(gpu_data.cusolverHandle, CUBLAS_FILL_MODE_LOWER, active_size,
                             gpu_data.d_A, active_size, gpu_data.d_work, gpu_data.lwork, gpu_data.d_info);
            CUDA_CHECK(cudaDeviceSynchronize());
            int info;
            CUDA_CHECK(cudaMemcpy(&info, gpu_data.d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (info != 0) {
                mexPrintf("Cholesky failed with info=%d\n", info);
                CUDA_CHECK(cudaMemcpy(x.data(), gpu_data.d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
                return x;
            }

            cusolverDnDpotrs(gpu_data.cusolverHandle, CUBLAS_FILL_MODE_LOWER, active_size, 1,
                             gpu_data.d_A, active_size, gpu_data.d_b, active_size, gpu_data.d_info);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 检查非负约束并更新 x
            bool all_nonnegative = true;
            std::vector<double> x_active(active_size);
            CUDA_CHECK(cudaMemcpy(x_active.data(), gpu_data.d_b, active_size * sizeof(double), cudaMemcpyDeviceToHost));
            size_t idx = 0;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    if (x_active[idx] < 0) {
                        all_nonnegative = false;
                        activeSet[j] = false;
                        bool false_val = false;
                        CUDA_CHECK(cudaMemcpy(&gpu_data.d_activeSet[j], &false_val, sizeof(bool), cudaMemcpyHostToDevice));
                    }
                    idx++;
                }
            }

            // 更新 x
            idx = 0;
            std::vector<double> x_new(n, 0.0);
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    x_new[j] = x_active[idx++];
                }
            }
            CUDA_CHECK(cudaMemcpy(gpu_data.d_x, x_new.data(), n * sizeof(double), cudaMemcpyHostToDevice));

            if (all_nonnegative) {
                break;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(x.data(), gpu_data.d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    return x;
}

extern "C" void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsActiveSetCUDA:input", "Two input arguments required: matrix C and vector d.");
    }

    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];
    size_t m = mxGetM(mxC);
    size_t n = mxGetN(mxC);

    std::vector<std::vector<double>> C(m, std::vector<double>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = mxGetPr(mxC)[i + j * m];
        }
    }

    std::vector<double> d(m);
    for (size_t i = 0; i < m; ++i) {
        d[i] = mxGetPr(mxD)[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> x = nnlsActiveSetCUDA(C, d);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = x[i];
    }

    mexPrintf(">> disp(x);\n");
    for (size_t i = 0; i < n; ++i) {
        if (x[i] == 0.0) {
            mexPrintf("         0\n");
        } else {
            mexPrintf("    %.4f\n", x[i]);
        }
    }
    mexPrintf("\nExecution Time: %lld microseconds\n", duration.count());
}
