#include "nnls.h"
#include <cstdio>
#include <chrono>

using namespace std::chrono;

// Main Function
int main() {
    // Example usage of lsqnonneg to match MATLAB interface
    int m = MATRIX_DIM; // Number of rows
    int n = MATRIX_DIM; // Number of columns
    float *h_C = new float[m * n];
    float *h_d = new float[n];
    float *h_x = new float[n];

    // Initialize h_C and h_d with random values
    srand(time(NULL));
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < n; ++i) {
        h_d[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Measure Execution Time and Call NNLS Solver
    auto start = high_resolution_clock::now();
    NNLS(h_C, h_d, h_x, m, n);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Active-Set Method Time: %ld microseconds\n", duration.count());

    // Print Active-Set Result
    printf("Active-Set Method Result:\n");
    for (int i = 0; i < n; ++i) {
        printf("%f ", h_x[i]);
        if ((i + 1) % MATRIX_DIM == 0) printf("\n");
    }

    // Clean up host memory
    delete[] h_C;
    delete[] h_d;
    delete[] h_x;

    return 0;
}
