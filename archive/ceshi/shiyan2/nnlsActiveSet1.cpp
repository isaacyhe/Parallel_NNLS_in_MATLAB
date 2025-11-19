#include "mex.h"  // MEX 文件必须包含的头文件
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <omp.h>  // OpenMP 头文件

using namespace std;

// 计算矩阵与向量的乘积（并行化）
vector<double> matVecMultiply(const vector<vector<double>>& mat, const vector<double>& vec) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<double> result(rows, 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += mat[i][j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

// 计算矩阵的转置（无需并行化）
vector<vector<double>> transpose(const vector<vector<double>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<vector<double>> transposed(cols, vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = mat[i][j];
        }
    }
    return transposed;
}

// 使用 Cholesky 分解求解线性方程组 Ax = b（部分并行化）
vector<double> choleskySolve(const vector<vector<double>>& A, const vector<double>& b) {
    size_t n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0.0));

    // Cholesky 分解
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (size_t k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    // 前向替换：解 Ly = b
    vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    // 后向替换：解 L^T x = y
    vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t j = i + 1; j < n; ++j) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    return x;
}

// 主动集法求解非负最小二乘问题（部分并行化）
vector<double> nnlsActiveSet(const vector<vector<double>>& C, const vector<double>& d, double tol = 1e-6) {
    size_t m = C.size();
    size_t n = C[0].size();

    // 初始化
    vector<double> x(n, 0.0);          // 变量 x，初始为 0
    vector<bool> activeSet(n, false);  // 活动集，初始为空
    vector<double> gradient(n, 0.0);   // 梯度

    // 外循环，直到满足收敛条件
    while (true) {
        // 计算残差 r = d - Cx
        vector<double> r = d;
        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += C[i][j] * x[j];
            }
            r[i] -= sum;
        }

        // 计算梯度 g = C^T r
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < m; ++i) {
                sum += C[i][j] * r[i];
            }
            gradient[j] = sum;
        }

        // 选择最大梯度的非活动变量
        int j_max = -1;
        double max_gradient = -numeric_limits<double>::infinity();
        for (size_t j = 0; j < n; ++j) {
            if (!activeSet[j] && gradient[j] > max_gradient) {
                max_gradient = gradient[j];
                j_max = j;
            }
        }

        // 如果最大梯度小于容忍度，结束算法
        if (max_gradient < tol) {
            break;
        }

        // 将该变量加入活动集
        activeSet[j_max] = true;

        // 内循环，处理活动集中的变量
        while (true) {
            // 构造活动集矩阵
            vector<vector<double>> C_active(m, vector<double>());
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    for (size_t i = 0; i < m; ++i) {
                        C_active[i].push_back(C[i][j]);
                    }
                }
            }

            // 计算 C_active^T C_active
            size_t active_size = C_active[0].size();
            vector<vector<double>> A(active_size, vector<double>(active_size, 0.0));
            #pragma omp parallel for
            for (size_t i = 0; i < active_size; ++i) {
                for (size_t j = 0; j < active_size; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < m; ++k) {
                        sum += C_active[k][i] * C_active[k][j];
                    }
                    A[i][j] = sum;
                }
            }

            // 计算 C_active^T d
            vector<double> b(active_size, 0.0);
            #pragma omp parallel for
            for (size_t i = 0; i < active_size; ++i) {
                double sum = 0.0;
                for (size_t k = 0; k < m; ++k) {
                    sum += C_active[k][i] * d[k];
                }
                b[i] = sum;
            }

            // 求解 A x = b
            vector<double> x_active = choleskySolve(A, b);

            // 更新 x
            size_t idx = 0;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j]) {
                    x[j] = x_active[idx++];
                } else {
                    x[j] = 0.0;
                }
            }

            // 检查非负约束
            bool all_nonnegative = true;
            for (size_t j = 0; j < n; ++j) {
                if (activeSet[j] && x[j] < 0) {
                    all_nonnegative = false;
                    activeSet[j] = false;  // 将负值变量移出活动集
                }
            }

            if (all_nonnegative) {
                break;
            }
        }
    }

    return x;
}

// MEX 文件入口函数
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 检查输入参数数量
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("nnlsActiveSet:input", "需要两个输入参数：矩阵 C 和向量 d。");
    }

    // 从 MATLAB 输入中提取矩阵 C 和向量 d
    const mxArray *mxC = prhs[0];
    const mxArray *mxD = prhs[1];

    size_t m = mxGetM(mxC);  // 矩阵 C 的行数
    size_t n = mxGetN(mxC);  // 矩阵 C 的列数

    // 将 MATLAB 的矩阵 C 转换为 C++ 的 vector<vector<double>>
    vector<vector<double>> C(m, vector<double>(n));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i][j] = mxGetPr(mxC)[i + j * m];
        }
    }

    // 将 MATLAB 的向量 d 转换为 C++ 的 vector<double>
    vector<double> d(m);
    for (size_t i = 0; i < m; ++i) {
        d[i] = mxGetPr(mxD)[i];
    }

    // 调用 nnlsActiveSet 函数
    vector<double> x = nnlsActiveSet(C, d);

    // 将结果转换回 MATLAB 的 mxArray
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    for (size_t i = 0; i < n; ++i) {
        mxGetPr(plhs[0])[i] = x[i];
    }
}