# Build Instructions

This document provides detailed instructions for building the NNLS solvers for different platforms and configurations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building CPU Implementations (OpenMP)](#building-cpu-implementations-openmp)
- [Building GPU Implementations (CUDA)](#building-gpu-implementations-cuda)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)

## Prerequisites

### General Requirements
- **MATLAB** R2018b or later (R2018a API compatibility)
- Sufficient RAM (8GB+ recommended for large problems)

### For CPU (OpenMP) Builds
- C++ compiler with C++11 support and OpenMP:
  - **Linux**: GCC 5.0+ or Clang 3.9+
  - **macOS**: Clang with OpenMP (via Homebrew) or GCC
  - **Windows**: MSVC 2017+ or MinGW-w64

### For GPU (CUDA) Builds
- **NVIDIA CUDA Toolkit** 11.0 or later
- **NVIDIA GPU** with compute capability 3.5+
- **cuBLAS** library (included with CUDA Toolkit)
- Compatible NVIDIA driver

## Building CPU Implementations (OpenMP)

### Linux/macOS

1. **Open MATLAB and navigate to the CPU source directory:**
   ```matlab
   cd src/cpu/
   ```

2. **Compile Active-Set solver (FP64):**
   ```matlab
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp' nnls_active_set_fp64_omp.cpp
   ```

3. **Compile Active-Set solver (FP32):**
   ```matlab
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp' nnls_active_set_fp32_omp.cpp
   ```

4. **Compile Gradient Projection solver (FP64):**
   ```matlab
   mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp' nnls_gradient_projection_fp64_omp.cpp
   ```

### Windows (MSVC)

1. **Set up MSVC compiler in MATLAB:**
   ```matlab
   mex -setup C++
   ```

2. **Navigate to CPU source directory:**
   ```matlab
   cd src/cpu/
   ```

3. **Compile with OpenMP support:**
   ```matlab
   mex -R2018a COMPFLAGS='$COMPFLAGS /openmp /O2' nnls_active_set_fp64_omp.cpp
   mex -R2018a COMPFLAGS='$COMPFLAGS /openmp /O2' nnls_active_set_fp32_omp.cpp
   mex -R2018a COMPFLAGS='$COMPFLAGS /openmp /O2' nnls_gradient_projection_fp64_omp.cpp
   ```

### macOS with Homebrew OpenMP

If using Apple Clang, you need to install OpenMP support:

```bash
brew install libomp
```

Then compile in MATLAB:
```matlab
cd src/cpu/
mex -R2018a CXXFLAGS='$CXXFLAGS -Xpreprocessor -fopenmp -O3' ...
    LDFLAGS='$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp' ...
    CXXOPTIMFLAGS='-O3 -DNDEBUG' ...
    nnls_active_set_fp64_omp.cpp
```

## Building GPU Implementations (CUDA)

### Prerequisites Check

Verify CUDA installation:
```bash
nvcc --version
nvidia-smi
```

### Linux

1. **Navigate to GPU source directory in MATLAB:**
   ```matlab
   cd src/gpu/
   ```

2. **Compile CUDA implementation:**
   ```matlab
   mex -R2018a nnls_gradient_projection_fp32_cuda.cu -lcublas
   ```

3. **With custom CUDA path (if needed):**
   ```matlab
   setenv('CUDA_PATH', '/usr/local/cuda-11.8');
   mex -R2018a -v nnls_gradient_projection_fp32_cuda.cu -lcublas
   ```

### Windows

1. **Ensure CUDA Toolkit is in PATH:**
   ```cmd
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   ```

2. **Compile in MATLAB:**
   ```matlab
   cd src/gpu/
   mex -R2018a nnls_gradient_projection_fp32_cuda.cu -lcublas
   ```

### macOS

CUDA support on macOS has been discontinued by NVIDIA. For GPU acceleration on macOS, consider:
- Using Metal Performance Shaders (requires code adaptation)
- Running on a Linux/Windows machine
- Using cloud GPU services

## Build Automation

### MATLAB Build Script

Create `build/build_all.m`:

```matlab
%% Build All NNLS Solvers
fprintf('Building NNLS solvers...\n');

% Navigate to source root
cd(fileparts(mfilename('fullpath')));
cd ..;

% Build CPU implementations
fprintf('\n=== Building CPU implementations ===\n');
cd src/cpu/

try
    fprintf('Building nnls_active_set_fp64_omp...\n');
    mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp' nnls_active_set_fp64_omp.cpp
    fprintf('✓ nnls_active_set_fp64_omp built successfully\n');
catch ME
    fprintf('✗ Failed to build nnls_active_set_fp64_omp: %s\n', ME.message);
end

try
    fprintf('Building nnls_active_set_fp32_omp...\n');
    mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp' nnls_active_set_fp32_omp.cpp
    fprintf('✓ nnls_active_set_fp32_omp built successfully\n');
catch ME
    fprintf('✗ Failed to build nnls_active_set_fp32_omp: %s\n', ME.message);
end

try
    fprintf('Building nnls_gradient_projection_fp64_omp...\n');
    mex -R2018a CXXFLAGS='$CXXFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp' nnls_gradient_projection_fp64_omp.cpp
    fprintf('✓ nnls_gradient_projection_fp64_omp built successfully\n');
catch ME
    fprintf('✗ Failed to build nnls_gradient_projection_fp64_omp: %s\n', ME.message);
end

cd ../..

% Build GPU implementations (if CUDA available)
fprintf('\n=== Building GPU implementations ===\n');
cd src/gpu/

try
    fprintf('Building nnls_gradient_projection_fp32_cuda...\n');
    mex -R2018a nnls_gradient_projection_fp32_cuda.cu -lcublas
    fprintf('✓ nnls_gradient_projection_fp32_cuda built successfully\n');
catch ME
    fprintf('✗ Failed to build CUDA implementation: %s\n', ME.message);
    fprintf('  (CUDA Toolkit may not be installed)\n');
end

cd ../..

fprintf('\n=== Build complete ===\n');
```

Run with:
```matlab
run('build/build_all.m');
```

## Platform-Specific Instructions

### Ubuntu/Debian Linux

Install prerequisites:
```bash
sudo apt-get update
sudo apt-get install build-essential
# For CUDA:
# Follow NVIDIA CUDA installation guide for your Ubuntu version
```

### CentOS/RHEL

```bash
sudo yum groupinstall "Development Tools"
# For CUDA:
# Follow NVIDIA CUDA installation guide for your RHEL version
```

### Windows with MinGW

1. Install MinGW-w64 with OpenMP support
2. Configure MATLAB to use MinGW:
   ```matlab
   mex -setup C++
   ```
3. Build as described in Windows section above

## Troubleshooting

### OpenMP Not Found

**Linux/macOS:**
```bash
# Install GCC with OpenMP
sudo apt-get install gcc g++  # Ubuntu/Debian
brew install gcc               # macOS
```

**Windows:**
- Ensure you're using MSVC 2017+ which includes OpenMP
- Or use MinGW-w64 with OpenMP support

### CUDA Not Found

Check CUDA installation:
```bash
which nvcc
echo $CUDA_PATH
```

Set CUDA path in MATLAB:
```matlab
setenv('CUDA_PATH', '/usr/local/cuda');
setenv('PATH', [getenv('CUDA_PATH') '/bin:' getenv('PATH')]);
```

### MEX Compiler Not Configured

```matlab
mex -setup C++
```

Select your preferred compiler from the list.

### OpenMP Runtime Warning

If you see "OMP: Warning #181: OMP_SET_NUM_THREADS: ignored", ensure:
- OpenMP is properly linked
- The number of threads is set correctly
- Your CPU supports the requested thread count

### CUDA Out of Memory

For large problems on GPU:
- Reduce problem size
- Use FP32 instead of FP64 (saves memory)
- Consider unified memory variants (future implementation)

## Testing

After building, test the implementations:

```matlab
% Test CPU implementation
cd examples/
C = rand(1000, 500);
d = rand(1000, 1);
x = nnls_active_set_fp64_omp(C, d, 4);
fprintf('Solution residual: %.6e\n', norm(C*x - d));
fprintf('Non-negativity check: %d\n', all(x >= 0));

% Test GPU implementation (if built)
try
    x_gpu = nnls_gradient_projection_fp32_cuda(single(C), single(d));
    fprintf('GPU residual: %.6e\n', norm(C*double(x_gpu) - d));
catch ME
    fprintf('GPU test failed: %s\n', ME.message);
end
```

## Performance Optimization

### OpenMP Thread Count

Set optimal thread count based on your CPU:
```matlab
num_threads = feature('numcores');  % Use all physical cores
x = nnls_active_set_fp64_omp(C, d, num_threads);
```

### Compiler Optimization Flags

For maximum performance, use optimization flags:
- `-O3` or `/O2`: Aggressive optimization
- `-march=native`: CPU-specific optimizations (GCC/Clang)
- `-ffast-math`: Relaxed floating-point (use with caution)

## Clean Build

To remove all MEX binaries:

```bash
# Linux/macOS
find . -name "*.mex*" -type f -delete

# Windows (PowerShell)
Get-ChildItem -Recurse -Filter "*.mex*" | Remove-Item
```

Or in MATLAB:
```matlab
delete('src/cpu/*.mex*');
delete('src/gpu/*.mex*');
```

## Additional Resources

- [MATLAB MEX Documentation](https://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html)
- [OpenMP Documentation](https://www.openmp.org/specifications/)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

## Support

For build issues:
1. Check the troubleshooting section above
2. Verify prerequisites are installed
3. Open an issue on GitHub with:
   - Platform and OS version
   - MATLAB version
   - Compiler version
   - Full error message
