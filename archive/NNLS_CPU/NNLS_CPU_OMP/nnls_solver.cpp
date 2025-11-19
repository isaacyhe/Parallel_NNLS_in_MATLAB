#include "mex.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

#define MATRIX_DIM 300
#define MAX_ITER_NNLS 1000
#define INITIAL_LEARNING_RATE 0.0005

void computeGradient(const std::vector<double> &A, const std::vector<double> &x, const std::vector<double> &b, std::vector<double> &gradient, int matrix_dim) {
    gradient.assign(matrix_dim, 0.0);
    #pragma omp parallel for
    for (int j = 0; j < matrix_dim; ++j) {
        double sum = 0.0;
        for (int k = 0; k < matrix_dim; ++k) {
            sum += A[k * matrix_dim + j] * x[k];
        }
        gradient[j] = sum - b[j];
    }
}

void updateX(std::vector<double> &x, const std::vector<double> &gradient, double learning_rate, int matrix_dim) {
    #pragma omp parallel for
    for (int j = 0; j < matrix_dim; ++j) {
        double old_xj = x[j];
        x[j] = std::max(0.0, old_xj - learning_rate * gradient[j]);
    }
}

void solveNNLS_GradientDescent_OpenMP(const std::vector<double> &A, std::vector<double> &x, const std::vector<double> &b, int matrix_dim) {
    std::vector<double> gradient(matrix_dim, 0.0);
    double learning_rate = INITIAL_LEARNING_RATE;

    for (int iter = 0; iter < MAX_ITER_NNLS; ++iter) {
        computeGradient(A, x, b, gradient, matrix_dim);
        updateX(x, gradient, learning_rate, matrix_dim);
    }
}

// MEX entry point
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check inputs
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:nnls_solver:invalidNumInputs", "Two inputs required: matrix C and vector d.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:nnls_solver:invalidNumOutputs", "One output required.");
    }

    // Get input matrix C and vector d
    double *C = mxGetPr(prhs[0]);
    double *d = mxGetPr(prhs[1]);
    int m = mxGetM(prhs[0]);  // Number of rows of C
    int n = mxGetN(prhs[0]);  // Number of columns of C

    // Initialize x with zeros
    std::vector<double> x(n, 0.0);

    // Solve NNLS using gradient descent
    solveNNLS_GradientDescent_OpenMP(std::vector<double>(C, C + m * n), x, std::vector<double>(d, d + m), n);

    // Create output array
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *output = mxGetPr(plhs[0]);

    // Copy result to output array
    for (int i = 0; i < n; ++i) {
        output[i] = x[i];
    }
}
