%% Build All NNLS Solvers
% Automated build script for all CPU and GPU implementations
%
% Usage: run('build/build_all.m') from project root
%
% Author: Parallel NNLS Team
% Date: 2025
% License: MIT

fprintf('====================================\n');
fprintf('   Building NNLS Solvers\n');
fprintf('====================================\n\n');

% Store original directory
orig_dir = pwd;

% Navigate to project root
cd(fileparts(mfilename('fullpath')));
cd ..;
project_root = pwd;

% Initialize build status
build_status = struct();

%% Build CPU implementations
fprintf('=== Building CPU implementations (OpenMP) ===\n\n');
cd(fullfile(project_root, 'src', 'cpu'));

% Detect platform for compiler flags
if ispc
    % Windows (MSVC)
    omp_flags = 'COMPFLAGS="$COMPFLAGS /openmp /O2"';
elseif ismac
    % macOS (may need Homebrew OpenMP)
    omp_flags = 'CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -O3" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp"';
else
    % Linux (GCC/Clang)
    omp_flags = 'CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp"';
end

% Build Active-Set FP64
try
    fprintf('Building nnls_active_set_fp64_omp...\n');
    eval(['mex -R2018a ' omp_flags ' nnls_active_set_fp64_omp.cpp']);
    fprintf('✓ nnls_active_set_fp64_omp built successfully\n\n');
    build_status.active_set_fp64 = true;
catch ME
    fprintf('✗ Failed to build nnls_active_set_fp64_omp\n');
    fprintf('  Error: %s\n\n', ME.message);
    build_status.active_set_fp64 = false;
end

% Build Active-Set FP32
try
    fprintf('Building nnls_active_set_fp32_omp...\n');
    eval(['mex -R2018a ' omp_flags ' nnls_active_set_fp32_omp.cpp']);
    fprintf('✓ nnls_active_set_fp32_omp built successfully\n\n');
    build_status.active_set_fp32 = true;
catch ME
    fprintf('✗ Failed to build nnls_active_set_fp32_omp\n');
    fprintf('  Error: %s\n\n', ME.message);
    build_status.active_set_fp32 = false;
end

% Build Gradient Projection FP64
try
    fprintf('Building nnls_gradient_projection_fp64_omp...\n');
    eval(['mex -R2018a ' omp_flags ' nnls_gradient_projection_fp64_omp.cpp']);
    fprintf('✓ nnls_gradient_projection_fp64_omp built successfully\n\n');
    build_status.grad_proj_fp64 = true;
catch ME
    fprintf('✗ Failed to build nnls_gradient_projection_fp64_omp\n');
    fprintf('  Error: %s\n\n', ME.message);
    build_status.grad_proj_fp64 = false;
end

% Build Gradient Projection FP32
try
    fprintf('Building nnls_gradient_projection_fp32_omp...\n');
    eval(['mex -R2018a ' omp_flags ' nnls_gradient_projection_fp32_omp.cpp']);
    fprintf('✓ nnls_gradient_projection_fp32_omp built successfully\n\n');
    build_status.grad_proj_fp32 = true;
catch ME
    fprintf('✗ Failed to build nnls_gradient_projection_fp32_omp\n');
    fprintf('  Error: %s\n\n', ME.message);
    build_status.grad_proj_fp32 = false;
end

%% Build GPU implementations
fprintf('=== Building GPU implementations (CUDA) ===\n\n');
cd(fullfile(project_root, 'src', 'gpu'));

% Check if CUDA is available
cuda_available = ~isempty(getenv('CUDA_PATH')) || exist('/usr/local/cuda', 'dir');

if ~cuda_available
    fprintf('⚠ CUDA Toolkit not detected, skipping GPU builds\n');
    fprintf('  Set CUDA_PATH environment variable if CUDA is installed\n\n');
    build_status.active_set_fp64_cuda = false;
    build_status.active_set_fp32_cuda = false;
    build_status.grad_proj_fp64_cuda = false;
    build_status.grad_proj_fp32_cuda = false;
else
    % Build Active-Set FP64 CUDA
    try
        fprintf('Building nnls_active_set_fp64_cuda...\n');
        mex -R2018a nnls_active_set_fp64_cuda.cu -lcublas
        fprintf('✓ nnls_active_set_fp64_cuda built successfully\n\n');
        build_status.active_set_fp64_cuda = true;
    catch ME
        fprintf('✗ Failed to build nnls_active_set_fp64_cuda\n');
        fprintf('  Error: %s\n\n', ME.message);
        fprintf('  Note: Ensure CUDA Toolkit and compatible GPU driver are installed\n\n');
        build_status.active_set_fp64_cuda = false;
    end

    % Build Active-Set FP32 CUDA
    try
        fprintf('Building nnls_active_set_fp32_cuda...\n');
        mex -R2018a nnls_active_set_fp32_cuda.cu -lcublas
        fprintf('✓ nnls_active_set_fp32_cuda built successfully\n\n');
        build_status.active_set_fp32_cuda = true;
    catch ME
        fprintf('✗ Failed to build nnls_active_set_fp32_cuda\n');
        fprintf('  Error: %s\n\n', ME.message);
        fprintf('  Note: Ensure CUDA Toolkit and compatible GPU driver are installed\n\n');
        build_status.active_set_fp32_cuda = false;
    end

    % Build Gradient Projection FP64 CUDA
    try
        fprintf('Building nnls_gradient_projection_fp64_cuda...\n');
        mex -R2018a nnls_gradient_projection_fp64_cuda.cu -lcublas
        fprintf('✓ nnls_gradient_projection_fp64_cuda built successfully\n\n');
        build_status.grad_proj_fp64_cuda = true;
    catch ME
        fprintf('✗ Failed to build nnls_gradient_projection_fp64_cuda\n');
        fprintf('  Error: %s\n\n', ME.message);
        fprintf('  Note: Ensure CUDA Toolkit and compatible GPU driver are installed\n\n');
        build_status.grad_proj_fp64_cuda = false;
    end

    % Build Gradient Projection FP32 CUDA
    try
        fprintf('Building nnls_gradient_projection_fp32_cuda...\n');
        mex -R2018a nnls_gradient_projection_fp32_cuda.cu -lcublas
        fprintf('✓ nnls_gradient_projection_fp32_cuda built successfully\n\n');
        build_status.grad_proj_fp32_cuda = true;
    catch ME
        fprintf('✗ Failed to build nnls_gradient_projection_fp32_cuda\n');
        fprintf('  Error: %s\n\n', ME.message);
        fprintf('  Note: Ensure CUDA Toolkit and compatible GPU driver are installed\n\n');
        build_status.grad_proj_fp32_cuda = false;
    end
end

%% Summary
fprintf('====================================\n');
fprintf('   Build Summary\n');
fprintf('====================================\n\n');

fprintf('CPU Implementations (OpenMP):\n');
fprintf('  %-40s %s\n', 'Active-Set FP64:', status_str(build_status.active_set_fp64));
fprintf('  %-40s %s\n', 'Active-Set FP32:', status_str(build_status.active_set_fp32));
fprintf('  %-40s %s\n', 'Gradient Projection FP64:', status_str(build_status.grad_proj_fp64));
fprintf('  %-40s %s\n', 'Gradient Projection FP32:', status_str(build_status.grad_proj_fp32));

fprintf('\nGPU Implementations (CUDA):\n');
fprintf('  %-40s %s\n', 'Active-Set FP64:', status_str(build_status.active_set_fp64_cuda));
fprintf('  %-40s %s\n', 'Active-Set FP32:', status_str(build_status.active_set_fp32_cuda));
fprintf('  %-40s %s\n', 'Gradient Projection FP64:', status_str(build_status.grad_proj_fp64_cuda));
fprintf('  %-40s %s\n', 'Gradient Projection FP32:', status_str(build_status.grad_proj_fp32_cuda));

fprintf('\n');
total = structfun(@(x) x, build_status);
success_count = sum(total);
total_count = length(total);

fprintf('Total: %d/%d successful\n\n', success_count, total_count);

if success_count == total_count
    fprintf('✓ All builds completed successfully!\n\n');
elseif success_count == 0
    fprintf('✗ All builds failed. Check compiler and OpenMP installation.\n');
    fprintf('  See BUILD.md for detailed instructions.\n\n');
else
    fprintf('⚠ Partial build success. Some implementations failed.\n');
    fprintf('  See BUILD.md for troubleshooting.\n\n');
end

% Return to original directory
cd(orig_dir);

fprintf('====================================\n\n');

%% Helper function
function str = status_str(success)
    if success
        str = '✓ SUCCESS';
    else
        str = '✗ FAILED';
    end
end
