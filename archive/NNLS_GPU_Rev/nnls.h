// headers.h (Updated for CUDA 12)
#ifndef HEADERS_H
#define HEADERS_H

// Includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// Define Constants (Flexible Dimensions)
#define GPU_USE 0                 // GPU selection
#define NSYS 1                    // Number of systems to solve (can be adjusted)
#define MAX_ITER_NNLS 1000        // Maximum number of update steps
#define MAX_ITER_LS 500           // Maximum number of update+downdate steps
#define TOL_TERMINATION 1e-6f     // Tolerance for termination
#define TOL_0 1e-6f               // Tolerance for downdate step

// Define sizes (make dimensions flexible)
const int MATRIX_DIM = 160;       // Replace fixed size with const variable

void NNLS(float *h_C, float *h_d, float *h_x, int m, int n);

#endif // HEADERS_H

