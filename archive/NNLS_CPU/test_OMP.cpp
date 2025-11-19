#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include "mex.h"

#define MAX_ITER_NNLS 1000

// NNLS Solver using OpenMP (modified for column-major storage)
void solveNNLS_ActiveSet_OpenMP(const std::vector<double> &A, std::vector<double> &x, const std::vector<double> &b, int matrix_dim) {
    std::vector<double> R(matrix_dim, 0.0);
    x.assign(matrix_dim, 0.0);

    bool isOptimal = false;
    for (int iter = 0; iter < MAX_ITER_NNLS && !isOptimal; ++iter) {
        // Compute A' * (b - A * x), adjusted for column-major storage
        #pragma omp parallel for
        for (int j = 0; j < matrix_dim; ++j) {
            double r = 0.0;
            for (int k = 0; k < matrix_dim; ++k) {
                r += A[j * matrix_dim + k] * (b[k] - A[j * matrix_dim + k] * x[k]);
            }
            R[j] = r;
        }

        // Update x based on active set
        isOptimal = true;
        #pragma omp parallel for reduction(&& : isOptimal)
        for (int j = 0; j < matrix_dim; ++j) {
            if (R[j] > 0) {
                x[j] += R[j];
                isOptimal = false;
            }
            x[j] = std::max(0.0, x[j]);
        }
    }
}

// MEX Function Interface
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check the input/output arguments
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:invalidNumInputs", "Two inputs required (matrix and vector).");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:lsqnonneg:invalidNumOutputs", "One output required (solution vector).");
    }

    // Get the matrix C (m x n) and vector d (m)
    double *C = mxGetPr(prhs[0]);
    double *d = mxGetPr(prhs[1]);
    
    // Get the dimensions of C and d
    mwSize m = mxGetM(prhs[0]);
    mwSize n = mxGetN(prhs[0]);

    // Initialize solution vector x
    std::vector<double> x(n, 0.0);

    // Call the NNLS solver
    solveNNLS_ActiveSet_OpenMP(std::vector<double>(C, C + m * n), x, std::vector<double>(d, d + m), n);

    // Create the output array for the solution vector
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *out = mxGetPr(plhs[0]);

    // Copy the solution into the output array
    for (mwSize i = 0; i < n; ++i) {
        out[i] = x[i];
    }
}
