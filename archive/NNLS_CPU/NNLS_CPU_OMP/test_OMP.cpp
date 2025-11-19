// nnls_openmp.cpp - Translation of NNLS CUDA code to modern OpenMP implementation
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

#define MATRIX_DIM 160
#define MAX_ITER_NNLS 1000

// NNLS Solver using OpenMP
void solveNNLS_ActiveSet_OpenMP(const std::vector<double> &A, std::vector<double> &x, const std::vector<double> &b, int matrix_dim) {
    std::vector<double> R(matrix_dim, 0.0);
    x.assign(matrix_dim, 0.0);

    bool isOptimal = false;
    for (int iter = 0; iter < MAX_ITER_NNLS && !isOptimal; ++iter) {
        // Compute A' * (b - A * x)
        #pragma omp parallel for
        for (int j = 0; j < matrix_dim; ++j) {
            double r = 0.0;
            for (int k = 0; k < matrix_dim; ++k) {
                r += A[k * matrix_dim + j] * (b[k] - A[k * matrix_dim + j] * x[j]);
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

// MATLAB-like Interface Function for lsqnonneg
std::vector<double> lsqnonneg(const std::vector<double> &C, const std::vector<double> &d) {
    int matrix_dim = MATRIX_DIM;
    std::vector<double> x(matrix_dim, 0.0);
    solveNNLS_ActiveSet_OpenMP(C, x, d, matrix_dim);
    return x;
}

// Main function to run NNLS solver
int main() {
    int m = MATRIX_DIM; // Number of rows
    int n = MATRIX_DIM; // Number of columns

    // Initialize input matrix and vector with random values
    std::vector<double> C(m * n);
    std::vector<double> d(m);

    // Seed for reproducibility
    srand(time(nullptr));
    for (int i = 0; i < m * n; ++i) {
        C[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < m; ++i) {
        d[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Measure execution time and solve NNLS
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> x = lsqnonneg(C, d);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("Execution Time: %ld microseconds\n", duration.count());

    // Print result
    printf("Result:\n");
    for (int i = 0; i < n; ++i) {
        printf("%f ", x[i]);
        if ((i + 1) % MATRIX_DIM == 0) printf("\n");
    }
    printf("\n");

    return 0;
}
