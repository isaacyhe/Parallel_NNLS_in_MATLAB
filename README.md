# Parallel NNLS Implementations

High-performance implementations of Non-Negative Least Squares (NNLS) solvers optimized for multi-core CPUs and GPUs, with applications to Magnetic Particle Imaging (MPI) reconstruction.

## Overview

This repository provides optimized implementations of NNLS solvers using:
- **Active-Set Algorithm** (Lawson & Hanson)
- **Gradient Projection Algorithm**

With support for:
- Multi-core CPU parallelization (OpenMP)
- GPU acceleration (CUDA)
- Single precision (FP32) and double precision (FP64)
- MATLAB MEX interface

The solvers find **x** that minimizes **||Cx - d||²** subject to **x ≥ 0**.

## Reference

This implementation is based on the research presented in:

> **Zhu et al. (2025).** "Accelerating Magnetic Particle Imaging with Data Parallelism: A Comparative Study"
> Proceedings of IEEE-MCSoC'25
> Available in: [`docs/MCSoC_25_ZHP.pdf`](docs/MCSoC_25_ZHP.pdf)

## Directory Structure

```
parallel_nnls_in_mpi/
├── src/                    # Source code
│   ├── cpu/               # OpenMP CPU implementations
│   │   ├── nnls_active_set_fp64_omp.cpp
│   │   ├── nnls_active_set_fp32_omp.cpp
│   │   ├── nnls_gradient_projection_fp64_omp.cpp
│   │   └── nnls_gradient_projection_fp32_omp.cpp
│   ├── gpu/               # CUDA GPU implementations
│   │   ├── nnls_active_set_fp64_cuda.cu
│   │   ├── nnls_active_set_fp32_cuda.cu
│   │   ├── nnls_gradient_projection_fp64_cuda.cu
│   │   └── nnls_gradient_projection_fp32_cuda.cu
│   ├── matlab/            # MATLAB scripts
│   │   └── NNLS_3D_Reconstruction.m
│   └── python/            # Python utilities
│       └── plot_nnls_performance.py
├── docs/                  # Documentation and papers
│   ├── MCSoC_25_ZHP.pdf
│   └── REFINEMENT_SUMMARY.md
├── examples/              # Usage examples
├── build/                 # Build system files
├── archive/               # Experimental code (archived)
├── README.md              # This file
└── BUILD.md               # Build instructions
```

## Features

### CPU Implementations (OpenMP)
- ✅ Active-Set method (FP32, FP64)
- ✅ Gradient Projection method (FP32, FP64)
- ✅ Optimized matrix operations
- ✅ Configurable thread count
- ✅ MATLAB MEX interface

### GPU Implementations (CUDA)
- ✅ Active-Set method (FP32, FP64)
- ✅ Gradient Projection method (FP32, FP64)
- ✅ Custom CUDA kernels for all operations
- ✅ Coalesced memory access patterns
- ✅ Optimized memory transfers
- ✅ Proper CUDA error checking

### Code Quality
- ✅ Comprehensive documentation
- ✅ MIT License
- ✅ English comments throughout
- ✅ Professional coding standards
- ✅ Performance optimizations

## Quick Start

### Prerequisites

**For CPU (OpenMP) implementations:**
- MATLAB (R2018b or later)
- C++ compiler with OpenMP support (GCC, Clang, or MSVC)

**For GPU (CUDA) implementations:**
- NVIDIA CUDA Toolkit (11.0+)
- Compatible NVIDIA GPU
- cuBLAS library

### Building MEX Files

1. **Navigate to source directory:**
   ```bash
   cd src/cpu/
   ```

2. **Compile in MATLAB:**
   ```matlab
   % Active-Set (FP64, OpenMP)
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp' nnls_active_set_fp64_omp.cpp

   % Active-Set (FP32, OpenMP)
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp' nnls_active_set_fp32_omp.cpp

   % Gradient Projection (FP64, OpenMP)
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp' nnls_gradient_projection_fp64_omp.cpp

   % Gradient Projection (FP32, OpenMP)
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp' nnls_gradient_projection_fp32_omp.cpp
   ```

3. **For CUDA implementations:**
   ```bash
   cd src/gpu/
   ```
   ```matlab
   % Active-Set (FP64, CUDA)
   mex -R2018a nnls_active_set_fp64_cuda.cu

   % Active-Set (FP32, CUDA)
   mex -R2018a nnls_active_set_fp32_cuda.cu

   % Gradient Projection (FP64, CUDA)
   mex -R2018a nnls_gradient_projection_fp64_cuda.cu

   % Gradient Projection (FP32, CUDA)
   mex -R2018a nnls_gradient_projection_fp32_cuda.cu
   ```

See [`BUILD.md`](BUILD.md) for detailed build instructions.

### Usage Example

```matlab
% Load or generate problem data
C = rand(1000, 500);      % Coefficient matrix
d = rand(1000, 1);        % Target vector
num_threads = 8;          % Number of OpenMP threads

% Solve NNLS using Active-Set method
x = nnls_active_set_fp64_omp(C, d, num_threads);

% Verify solution
residual = norm(C * x - d);
fprintf('Residual: %.6e\n', residual);
fprintf('Non-negativity satisfied: %d\n', all(x >= 0));
```

## Performance

The implementations have been optimized for:
- **Speed**: OpenMP parallelization and cuBLAS for matrix operations
- **Memory**: Efficient memory allocation and reduced GPU transfers
- **Precision**: Both FP32 and FP64 support
- **Scalability**: Tested on problems with dimensions up to 10000×5000

See [`docs/`](docs/) for performance benchmarks and analysis.

## Algorithms

### Active-Set Method
- Iteratively moves variables between active and passive sets
- Solves unconstrained least squares on passive variables
- Guaranteed convergence to global optimum
- Best for small to medium-sized problems

### Gradient Projection Method
- Projects gradient descent steps onto feasible region
- Fast convergence for well-conditioned problems
- Efficient for large-scale problems
- GPU-friendly algorithm structure

## Applications

Originally developed for **Magnetic Particle Imaging (MPI)** 3D reconstruction:
- Real-time image reconstruction
- Large-scale inverse problems
- Non-negative concentration constraints
- High-throughput data processing

Also applicable to:
- Signal processing
- Image deconvolution
- Spectral unmixing
- Machine learning (non-negative matrix factorization)

## Development History

This repository contains refined, production-ready code. Experimental implementations have been archived in `archive/` for reference. See [`docs/REFINEMENT_SUMMARY.md`](docs/REFINEMENT_SUMMARY.md) for details on improvements made.

## Contributing

Contributions are welcome! Please:
1. Follow the existing code style
2. Add comprehensive documentation
3. Include tests for new features
4. Update relevant documentation files

## License

MIT License - see individual source files for headers.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhu2025accelerating,
  title={Accelerating Magnetic Particle Imaging with Data Parallelism: A Comparative Study},
  author={Zhu, et al.},
  booktitle={Proceedings of the IEEE International Symposium on Multi-Core Systems-on-Chip (MCSoC'25)},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Based on the classical NNLS algorithm by Lawson & Hanson (1995)
- Developed for Magnetic Particle Imaging research
- Optimized for modern parallel computing architectures
