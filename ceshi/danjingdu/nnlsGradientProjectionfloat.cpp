#include "mex.h"
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <omp.h>

using namespace std;

// 计算矩阵A和向量x的乘积（优化后的版本）
vector<float> matVecMultiply(const vector<vector<float>>& A, const vector<float>& x) {
    size_t m = A.size();
    size_t n = A[0].size();
    vector<float> result(m, 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        result[i] = sum;
    }

    return result;
}

// 计算残差 r = Ax - b（优化后的版本）
vector<float> computeResidual(const vector<vector<float>>& A, const vector<float>& x, const vector<float>& b) {
    vector<float> Ax = matVecMultiply(A, x);
    vector<float> r(Ax.size(), 0.0f);

    #pragma omp parallel for
    for (size_t i = 0; i < Ax.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }

    return r;
}

// 计算向量的范数
float vectorNorm(const vector<float>& v) {
    float norm = 0.0f;

    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < v.size(); ++i) {
        norm += v[i] * v[i];
    }

    return sqrtf(norm);  // 使用 sqrtf 而不是 sqrt
}

// 梯度投影法：非负最小二乘法
vector<float> lsqnonnegGradientProjection(const vector<vector<float>>& A, const vector<float>& b, float tol = 1e-6f, int max_iter = 2000) {
    size_t m = A.size();
    size_t n = A[0].size();

    vector<float> x(n, 0.0f);  // 初始解 x = 0
    float alpha = 0.001f;      // 学习率（单精度浮点数）

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // 计算残差 r = Ax - b
        vector<float> r = computeResidual(A, x, b);

        // 计算梯度 g = A^T r
        vector<float> g(n, 0.0f);

        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < m; ++i) {
                sum += A[i][j] * r[i];
            }
            g[j] = sum;
        }

        // 使用梯度更新 x
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            x[j] -= alpha * g[j];
        }

        // 投影操作：将 x 投影到非负空间
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            if (x[j] < 0) {
                x[j] = 0;  // 投影到非负空间
            }
        }

        // 计算残差的范数
        float residualNorm = vectorNorm(r);

        // 如果残差范数小于容忍度，则认为优化完成
        if (residualNorm < tol) {
            mexPrintf("Converged after %d iterations.\n", iter);
            break;
        }
    }

    return x;
}

// MEX 文件入口函数
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 检查输入参数数量
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("nnlsGradientProjection:input", "需要三个输入参数：矩阵 A、向量 b 和线程数。");
    }

    // 从 MATLAB 输入中提取矩阵 A 和向量 b
    const mxArray *mxA = prhs[0];
    const mxArray *mxB = prhs[1];

    size_t m = mxGetM(mxA);  // 矩阵 A 的行数
    size_t n = mxGetN(mxA);  // 矩阵 A 的列数

    // 将 MATLAB 的矩阵 A 转换为 C++ 的 vector<vector<float>>
    vector<vector<float>> A(m, vector<float>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = static_cast<float>(mxGetPr(mxA)[i + j * m]);
        }
    }

    // 将 MATLAB 的向量 b 转换为 C++ 的 vector<float>
    vector<float> b(m);
    for (size_t i = 0; i < m; ++i) {
        b[i] = static_cast<float>(mxGetPr(mxB)[i]);
    }

    // 获取线程数参数
    int num_threads = static_cast<int>(mxGetScalar(prhs[2]));
    omp_set_num_threads(num_threads);  // 设置 OpenMP 线程数

    // 记录程序开始时间
    auto start = chrono::high_resolution_clock::now();

    // 调用梯度投影法求解非负最小二乘问题
    vector<float> x = lsqnonnegGradientProjection(A, b);

    // 记录程序结束时间
    auto end = chrono::high_resolution_clock::now();

    // 计算运行时间
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // 将结果转换回 MATLAB 的 mxArray
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);  // MATLAB 默认使用 double
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = static_cast<double>(x[i]);  // 转换回 double 返回给 MATLAB
    }

    // 输出解向量 x，格式与 MATLAB 的 disp 一致
    mexPrintf(">> disp(x);\n");
    for (size_t i = 0; i < n; ++i) {
        if (x[i] == 0.0f) {
            mexPrintf("         0\n");  // 对齐 0 值
        } else {
            mexPrintf("    %.4f\n", x[i]);  // 保留 4 位小数
        }
    }

    // 输出运行时间
    mexPrintf("\nExecution Time: %lld microseconds\n", duration.count());
}
