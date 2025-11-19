#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cassert>
#include <chrono>
#include <omp.h>
#include "mex.h"
#include <cstdlib>
#include <ctime>

using namespace std;

// Function to perform matrix-vector multiplication
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

// Function to compute the residual vector r = Ax - b
vector<double> computeResidual(const vector<vector<double>>& A, const vector<double>& x, const vector<double>& b) {
    vector<double> Ax = matVecMultiply(A, x);
    vector<double> r(Ax.size(), 0.0);

    #pragma omp parallel for
    for (size_t i = 0; i < Ax.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }
    return r;
}

// Function to compute the Euclidean norm of a vector
double vectorNorm(const vector<double>& v) {
    double norm = 0.0;

    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < v.size(); ++i) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

// Gradient Descent with backtracking line search for Non-Negative Least Squares
vector<double> lsqnonneg(const vector<vector<double>>& A, const vector<double>& b, double tol = 1e-6, int max_iter = 10000) {
    size_t m = A.size();
    size_t n = A[0].size();
    vector<double> x(n, 0.0); // Initialize x to zeros
    double alpha = 1.0; // Initial step size
    double rho = 0.5; // Backtracking parameter
    double c = 0.1; // Sufficient decrease parameter

    for (int iter = 0; iter < max_iter; ++iter) {
        vector<double> r = computeResidual(A, x, b);
        vector<double> g(n, 0.0);

        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                g[j] += A[i][j] * r[i];
            }
        }

        // Backtracking line search
        double prev_norm = vectorNorm(r);
        vector<double> x_new = x;
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j) {
            x_new[j] = max(0.0, x[j] - alpha * g[j]);
        }
        vector<double> r_new = computeResidual(A, x_new, b);
        double new_norm = vectorNorm(r_new);

        while (new_norm > prev_norm - c * alpha * vectorNorm(g)) {
            alpha *= rho;
            #pragma omp parallel for
            for (size_t j = 0; j < n; ++j) {
                x_new[j] = max(0.0, x[j] - alpha * g[j]);
            }
            r_new = computeResidual(A, x_new, b);
            new_norm = vectorNorm(r_new);
        }
        x = x_new;

        if (new_norm < tol) {
            break;
        }
    }
    return x;
}

// MEX gateway function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    // Check for proper number of input and output arguments
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:nrhs", "Two inputs required: matrix A and vector b.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:nlhs", "Too many output arguments.");
    }

    // Get dimensions of input matrix A
    mwSize m = mxGetM(prhs[0]);
    mwSize n = mxGetN(prhs[0]);

    // Check if A is a real double matrix
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:notDouble", "Input matrix A must be a real double-precision matrix.");
    }

    // Check if b is a real double vector
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != m) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:notDoubleVector", "Input vector b must be a real double-precision vector with length equal to the number of rows in A.");
    }


    // Get pointers to input data
    double* A_data = mxGetPr(prhs[0]);
    double* b_data = mxGetPr(prhs[1]);

    // Convert input data to C++ vectors
    vector<vector<double>> A(m, vector<double>(n, 0.0));
    vector<double> b(m, 0.0);

    for (mwSize i = 0; i < m; ++i) {
        for (mwSize j = 0; j < n; ++j) {
            A[i][j] = A_data[i + m * j];
        }
        b[i] = b_data[i];
    }

    // Call the non-negative least squares solver
    vector<double> x = lsqnonneg(A, b);

    // Create output matrix
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double* x_data = mxGetPr(plhs[0]);

    // Copy the solution vector to the output matrix
    for (size_t i = 0; i < n; ++i) {
        x_data[i] = x[i];
    }
}