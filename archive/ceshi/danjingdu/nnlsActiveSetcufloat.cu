#include "mex.h"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        mexPrintf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        mexErrMsgIdAndTxt("CUDA:error", cudaGetErrorString(err)); \
    } \
} while (0)

// CUDA kernel for matrix-vector multiplication
__global__ void matVecMultiplyKernel(const float* mat, const float* vec, float* result, size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += mat[i * cols + j] * vec[j];
        }
        result[i] = sum;
    }
}

vector<float> matVecMultiplyCUDA(const vector<vector<float>>& mat, const vector<float>& vec) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<float> result(rows, 0.0f);

    float *d_mat, *d_vec, *d_result;
    CUDA_CHECK(cudaMalloc(&d_mat, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vec, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, rows * sizeof(float)));

    vector<float> flat_mat(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat_mat[i * cols + j] = mat[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_mat, flat_mat.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, vec.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    matVecMultiplyKernel<<<gridSize, blockSize>>>(d_mat, d_vec, d_result, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_result, rows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

// CUDA kernel for matrix transpose
__global__ void transposeKernel(const float* mat, float* transposed, size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        transposed[j * rows + i] = mat[i * cols + j];
    }
}

vector<vector<float>> transposeCUDA(const vector<vector<float>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<vector<float>> transposed(cols, vector<float>(rows));

    float *d_mat, *d_transposed;
    CUDA_CHECK(cudaMalloc(&d_mat, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transposed, rows * cols * sizeof(float)));

    vector<float> flat_mat(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flat_mat[i * cols + j] = mat[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_mat, flat_mat.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);
    transposeKernel<<<gridDim, blockDim>>>(d_mat, d_transposed, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> flat_transposed(rows * cols);
    CUDA_CHECK(cudaMemcpy(flat_transposed.data(), d_transposed, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            transposed[j][i] = flat_transposed[j * rows + i];
        }
    }

    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_transposed));

    return transposed;
}

// CPU-based Cholesky solve (for correctness)
vector<float> choleskySolve(const vector<vector<float>>& A, const vector<float>& b) {
    size_t n = A.size();
    vector<vector<float>> L(n, vector<float>(n, 0.0f));

    // Cholesky decomposition
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            float diag = A[i][i] - sum;
            if (i == j) {
                if (diag <= 0) {
                    mexPrintf("Matrix is not positive definite at i=%zu, diag=%f\n", i, diag);
                    return vector<float>(n, -1.0f); // Return error indicator
                }
                L[i][j] = sqrtf(diag);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    // Forward substitution: Ly = b
    vector<float> y(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // Backward substitution: L^T x = y
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

vector<float> nnlsActiveSetCUDA(const vector<vector<float>>& C, const vector<float>& d, float tol = 1e-6f) {
    size_t m = C.size();
    size_t n = C[0].size();

    vector<float> x(n, 0.0f);
    vector<bool> activeSet(n, false);
    vector<float> gradient(n, 0.0f);

    while (true) {
        vector<float> r = d;
        vector<float> Cx = matVecMultiplyCUDA(C, x);
        for (size_t i = 0; i < m; ++i) {
            r[i] -= Cx[i];
        }

        vector<vector<float>> Ct = transposeCUDA(C);
        gradient = matVecMultiplyCUDA(Ct, r);

        int j_max = -1;
        float max_gradient = -numeric_limits<float>::infinity();
        for (size_t j = 0; j < n; ++j) {
            if (!activeSet[j] && gradient[j] > max_gradient) {
                max_gradient = gradient[j];
                j_max = j;
            }
        }

        if (max_gradient < tol) {
            break;
        }

        activeSet[j_max] = true;

        while (true) {
            vector<vector<float>> C_active(m, vector<float>());
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    for (size_t i = 0; i < m; ++i) {
                        C_active[i].push_back(C[i][j]);
                    }
                }
            }

            size_t active_size = C_active[0].size();
            vector<vector<float>> A(active_size, vector<float>(active_size, 0.0f));
            vector<vector<float>> C_active_T = transposeCUDA(C_active);
            for (size_t i = 0; i < active_size; ++i) {
                for (size_t j = 0; j < active_size; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < m; ++k) {
                        sum += C_active[k][i] * C_active[k][j];
                    }
                    A[i][j] = sum;
                }
            }

            vector<float> b(active_size, 0.0f);
            for (size_t i = 0; i < active_size; ++i) {
                float sum = 0.0f;
                for (size_t k = 0; k < m; ++k) {
                    sum += C_active[k][i] * d[k];
                }
                b[i] = sum;
            }

            vector<float> x_active = choleskySolve(A, b);
            if (x_active[0] == -1.0f) { // Error from Cholesky
                mexPrintf("Cholesky failed, returning current x\n");
                return x;
            }

            size_t idx = 0;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    x[j] = x_active[idx++];
                } else {
                    x[j] = 0.0f;
                }
            }

            bool all_nonnegative = true;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j] && x[j] < 0) {
                    all_nonnegative = false;
                    activeSet[j] = false;
                }
            }

            if (all_nonnegative) {
                break;
            }
        }
    }

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

    vector<vector<float>> C(m, vector<float>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = (float)mxGetPr(mxC)[i + j * m];
        }
    }

    vector<float> d(m);
    for (size_t i = 0; i < m; ++i) {
        d[i] = (float)mxGetPr(mxD)[i];
    }

    auto start = chrono::high_resolution_clock::now();
    vector<float> x = nnlsActiveSetCUDA(C, d);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = x[i];
    }

    mexPrintf(">> disp(x);\n");
    for (size_t i = 0; i < n; ++i) {
        if (x[i] == 0.0f) {
            mexPrintf("         0\n");
        } else {
            mexPrintf("    %.4f\n", x[i]);
        }
    }
    mexPrintf("\nExecution Time: %lld microseconds\n", duration.count());
}
