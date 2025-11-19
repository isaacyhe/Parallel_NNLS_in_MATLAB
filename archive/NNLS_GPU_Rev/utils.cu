// utils.cu (Updated for CUDA 12)
#include "nnls.h"

// Utility Function for Error Handling
inline cudaError_t checkCuda(cudaError_t result, const char *fn) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error in " << fn << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

