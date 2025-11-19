#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <chrono>
#include <omp.h>  // 引入 OpenMP 库
#include "mex.h"  // 引入 MEX 库

using namespace std;

// 计算矩阵A和向量x的乘积
vector<double> matVecMultiply(const vector<vector<double>>& A, const vector<double>& x) {
    size_t m = A.size();
    size_t n = A[0].size();
    vector<double> result(m, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// 计算残差 r = Ax - b
vector<double> computeResidual(const vector<vector<double>>& A, const vector<double>& x, const vector<double>& b) {
    vector<double> Ax = matVecMultiply(A, x);
    vector<double> r(Ax.size(), 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < Ax.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }
    return r;
}

// 计算向量的范数
double vectorNorm(const vector<double>& v) {
    double norm = 0.0;

    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < v.size(); ++i) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

// 梯度下降法：非负最小二乘法
vector<double> lsqnonneg(const vector<vector<double>>& A, const vector<double>& b, double tol = 1e-6, double max_iter = 1000) {
    size_t m = A.size();
    size_t n = A[0].size();
    
    vector<double> x(n, 0.0);    // 初始解 x = 0
    double alpha = 0.001;  // 学习率（调小）

    // 用于检测收敛
    double prevResidualNorm = std::numeric_limits<double>::infinity();

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // 计算残差 r = Ax - b
        vector<double> r = computeResidual(A, x, b);
        
        // 计算梯度 g = A^T r
        vector<double> g(n, 0.0);
        
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                g[j] += A[i][j] * r[i];
            }
        }

        // 使用梯度更新 x，并保证每个元素非负
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            x[j] -= alpha * g[j];
            if (x[j] < 0) {
                x[j] = 0;  // 保证解是非负的
            }
        }

        // 计算当前残差的范数
        double residualNorm = vectorNorm(r);

        // 如果残差范数变化非常小，认为已经收敛
        if (fabs(prevResidualNorm - residualNorm) < tol) {
            break;
        }

        prevResidualNorm = residualNorm;
    }

    return x;
}

// MEX 接口函数
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // 确保输入参数是合法的
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:nrhs", "两输入参数：矩阵 A 和向量 b。");
    }

    // 解析输入参数
    double* A_data = mxGetPr(prhs[0]);  // 获取矩阵 A 的数据指针
    double* b_data = mxGetPr(prhs[1]);  // 获取向量 b 的数据指针

    mwSize m = mxGetM(prhs[0]);  // 获取矩阵 A 的行数
    mwSize n = mxGetN(prhs[0]);  // 获取矩阵 A 的列数

    // 将输入的矩阵 A 和向量 b 转换为 C++ 类型
    vector<vector<double>> A(m, vector<double>(n, 0.0));
    vector<double> b(m, 0.0);

    // 将矩阵 A 填充进 C++ 类型
    for (mwSize i = 0; i < m; ++i) {
        for (mwSize j = 0; j < n; ++j) {
            A[i][j] = A_data[i + m * j];
        }
    }

    // 将向量 b 填充进 C++ 类型
    for (mwSize i = 0; i < m; ++i) {
        b[i] = b_data[i];
    }

    // 调用非负最小二乘法求解
    vector<double> x = lsqnonneg(A, b);

    // 创建输出矩阵
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* x_data = mxGetPr(plhs[0]);

    // 将结果 x 填充到输出矩阵
    for (size_t i = 0; i < n; ++i) {
        x_data[i] = x[i];
    }
}
