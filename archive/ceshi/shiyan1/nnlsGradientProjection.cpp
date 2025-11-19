#include "mex.h"
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>

using namespace std;

// 计算矩阵A和向量x的乘积
vector<double> matVecMultiply(const vector<vector<double>>& A, const vector<double>& x) {
    size_t m = A.size();
    size_t n = A[0].size();
    vector<double> result(m, 0.0);
    
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
    
    for (size_t i = 0; i < Ax.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }
    return r;
}

// 计算向量的范数
double vectorNorm(const vector<double>& v) {
    double norm = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

// 梯度投影法：非负最小二乘法
vector<double> lsqnonnegGradientProjection(const vector<vector<double>>& A, const vector<double>& b, double tol = 1e-6, double max_iter = 10000) {
    size_t m = A.size();
    size_t n = A[0].size();
    
    vector<double> x(n, 0.0);    // 初始解 x = 0
    double alpha = 0.001;  // 学习率（调整为更小的值）
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
        // 计算残差 r = Ax - b
        vector<double> r = computeResidual(A, x, b);
        
        // 计算梯度 g = A^T r
        vector<double> g(n, 0.0);
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                g[j] += A[i][j] * r[i];
            }
        }

        // 使用梯度更新 x
        for (size_t j = 0; j < n; ++j) {
            x[j] -= alpha * g[j];
        }

        // 投影操作：将 x 投影到非负空间
        for (size_t j = 0; j < n; ++j) {
            if (x[j] < 0) {
                x[j] = 0;  // 投影到非负空间
            }
        }

        // 计算残差的范数
        double residualNorm = vectorNorm(r);
        
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
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsGradientProjection:input", "需要两个输入参数：矩阵 A 和向量 b。");
    }

    // 从 MATLAB 输入中提取矩阵 A 和向量 b
    const mxArray *mxA = prhs[0];
    const mxArray *mxB = prhs[1];

    size_t m = mxGetM(mxA);  // 矩阵 A 的行数
    size_t n = mxGetN(mxA);  // 矩阵 A 的列数

    // 将 MATLAB 的矩阵 A 转换为 C++ 的 vector<vector<double>>
    vector<vector<double>> A(m, vector<double>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i][j] = mxGetPr(mxA)[i + j * m];
        }
    }

    // 将 MATLAB 的向量 b 转换为 C++ 的 vector<double>
    vector<double> b(m);
    for (size_t i = 0; i < m; ++i) {
        b[i] = mxGetPr(mxB)[i];
    }

    // 调用梯度投影法求解非负最小二乘问题
    vector<double> x = lsqnonnegGradientProjection(A, b);

    // 将结果转换回 MATLAB 的 mxArray
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = x[i];
    }
}