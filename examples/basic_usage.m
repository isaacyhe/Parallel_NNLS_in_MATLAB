%% Basic Usage Example for NNLS Solvers
% This script demonstrates how to use the parallel NNLS implementations
%
% Prerequisites:
%   - Build all MEX files using build/build_all.m
%   - Add src/cpu/ and src/gpu/ to MATLAB path
%
% Author: Parallel NNLS Team
% Date: 2025
% License: MIT

%% Setup
clear; clc;
fprintf('NNLS Solver Basic Usage Example\n');
fprintf('================================\n\n');

% Add source directories to path
addpath('../src/cpu');
addpath('../src/gpu');

%% Generate Test Problem
fprintf('Generating test problem...\n');

% Problem dimensions
m = 2000;  % Number of rows (equations)
n = 1000;  % Number of columns (variables)

% Generate random problem: minimize ||Cx - d||^2 subject to x >= 0
rng(42);  % For reproducibility
C = randn(m, n);
x_true = abs(randn(n, 1));  % True non-negative solution
d = C * x_true + 0.01 * randn(m, 1);  % Add small noise

fprintf('  Problem size: C is %d×%d, d is %d×1\n', m, n, m);
fprintf('  True solution has %d non-zero elements\n\n', sum(x_true > 1e-10));

%% Solve using Active-Set (FP64, OpenMP)
fprintf('=== Active-Set Method (FP64, OpenMP) ===\n');

num_threads = 8;  % Adjust based on your CPU
fprintf('Using %d threads\n', num_threads);

try
    tic;
    x_active = nnls_active_set_fp64_omp(C, d, num_threads);
    time_active = toc;

    residual_active = norm(C * x_active - d);
    fprintf('Execution time: %.4f seconds\n', time_active);
    fprintf('Residual: %.6e\n', residual_active);
    fprintf('Non-negativity satisfied: %d\n', all(x_active >= -1e-10));
    fprintf('Number of non-zeros: %d\n\n', sum(x_active > 1e-10));
catch ME
    fprintf('✗ Failed: %s\n', ME.message);
    fprintf('  Make sure to build the MEX file first!\n\n');
end

%% Solve using Active-Set (FP32, OpenMP)
fprintf('=== Active-Set Method (FP32, OpenMP) ===\n');

try
    tic;
    x_active_fp32 = nnls_active_set_fp32_omp(single(C), single(d), num_threads);
    x_active_fp32 = double(x_active_fp32);  % Convert back to double for comparison
    time_active_fp32 = toc;

    residual_active_fp32 = norm(C * x_active_fp32 - d);
    fprintf('Execution time: %.4f seconds\n', time_active_fp32);
    fprintf('Residual: %.6e\n', residual_active_fp32);
    fprintf('Non-negativity satisfied: %d\n', all(x_active_fp32 >= -1e-6));
    fprintf('Number of non-zeros: %d\n\n', sum(x_active_fp32 > 1e-6));
catch ME
    fprintf('✗ Failed: %s\n', ME.message);
    fprintf('  Make sure to build the MEX file first!\n\n');
end

%% Solve using Gradient Projection (FP64, OpenMP)
fprintf('=== Gradient Projection Method (FP64, OpenMP) ===\n');

try
    tic;
    x_gradproj = nnls_gradient_projection_fp64_omp(C, d, num_threads);
    time_gradproj = toc;

    residual_gradproj = norm(C * x_gradproj - d);
    fprintf('Execution time: %.4f seconds\n', time_gradproj);
    fprintf('Residual: %.6e\n', residual_gradproj);
    fprintf('Non-negativity satisfied: %d\n', all(x_gradproj >= -1e-10));
    fprintf('Number of non-zeros: %d\n\n', sum(x_gradproj > 1e-10));
catch ME
    fprintf('✗ Failed: %s\n', ME.message);
    fprintf('  Make sure to build the MEX file first!\n\n');
end

%% Solve using GPU (if available)
fprintf('=== Gradient Projection Method (FP32, CUDA) ===\n');

try
    tic;
    x_gpu = nnls_gradient_projection_fp32_cuda(single(C), single(d));
    x_gpu = double(x_gpu);  % Convert back to double
    time_gpu = toc;

    residual_gpu = norm(C * x_gpu - d);
    fprintf('Execution time: %.4f seconds\n', time_gpu);
    fprintf('Residual: %.6e\n', residual_gpu);
    fprintf('Non-negativity satisfied: %d\n', all(x_gpu >= -1e-6));
    fprintf('Number of non-zeros: %d\n\n', sum(x_gpu > 1e-6));
catch ME
    fprintf('✗ GPU implementation not available: %s\n', ME.message);
    fprintf('  CUDA implementation may not be built or GPU not available\n\n');
end

%% Compare with MATLAB's built-in lsqnonneg (if available)
fprintf('=== MATLAB lsqnonneg (for comparison) ===\n');

try
    tic;
    x_matlab = lsqnonneg(C, d);
    time_matlab = toc;

    residual_matlab = norm(C * x_matlab - d);
    fprintf('Execution time: %.4f seconds\n', time_matlab);
    fprintf('Residual: %.6e\n', residual_matlab);
    fprintf('Number of non-zeros: %d\n\n', sum(x_matlab > 1e-10));
catch ME
    fprintf('✗ lsqnonneg not available: %s\n\n', ME.message);
end

%% Performance Summary
fprintf('================================\n');
fprintf('Performance Summary\n');
fprintf('================================\n');

try
    fprintf('Active-Set (FP64):        %.4f sec\n', time_active);
    fprintf('Active-Set (FP32):        %.4f sec (%.1fx)\n', time_active_fp32, time_active/time_active_fp32);
    fprintf('Gradient Proj (FP64):     %.4f sec (%.1fx)\n', time_gradproj, time_active/time_gradproj);
    if exist('time_gpu', 'var')
        fprintf('Gradient Proj (GPU):      %.4f sec (%.1fx)\n', time_gpu, time_active/time_gpu);
    end
    if exist('time_matlab', 'var')
        fprintf('MATLAB lsqnonneg:         %.4f sec (%.1fx)\n', time_matlab, time_active/time_matlab);
    end
catch
    fprintf('Run all implementations to see performance comparison\n');
end

fprintf('\n');

%% Visualization (optional)
fprintf('Generating solution comparison plot...\n');

figure('Position', [100, 100, 1200, 400]);

% Plot 1: Solution comparison
subplot(1, 3, 1);
if exist('x_active', 'var')
    plot(x_true, 'k-', 'LineWidth', 1.5); hold on;
    plot(x_active, 'r--', 'LineWidth', 1);
    legend('True', 'Active-Set');
    title('Solution Comparison');
    xlabel('Variable index');
    ylabel('Value');
    grid on;
end

% Plot 2: Residuals
subplot(1, 3, 2);
residuals = [];
methods = {};
if exist('residual_active', 'var')
    residuals(end+1) = residual_active;
    methods{end+1} = 'Active (FP64)';
end
if exist('residual_active_fp32', 'var')
    residuals(end+1) = residual_active_fp32;
    methods{end+1} = 'Active (FP32)';
end
if exist('residual_gradproj', 'var')
    residuals(end+1) = residual_gradproj;
    methods{end+1} = 'GradProj (FP64)';
end
if exist('residual_gpu', 'var')
    residuals(end+1) = residual_gpu;
    methods{end+1} = 'GPU (FP32)';
end
if ~isempty(residuals)
    bar(residuals);
    set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 45);
    title('Residual Comparison');
    ylabel('||Cx - d||');
    grid on;
end

% Plot 3: Execution times
subplot(1, 3, 3);
times = [];
time_methods = {};
if exist('time_active', 'var')
    times(end+1) = time_active;
    time_methods{end+1} = 'Active (FP64)';
end
if exist('time_active_fp32', 'var')
    times(end+1) = time_active_fp32;
    time_methods{end+1} = 'Active (FP32)';
end
if exist('time_gradproj', 'var')
    times(end+1) = time_gradproj;
    time_methods{end+1} = 'GradProj (FP64)';
end
if exist('time_gpu', 'var')
    times(end+1) = time_gpu;
    time_methods{end+1} = 'GPU (FP32)';
end
if ~isempty(times)
    bar(times);
    set(gca, 'XTickLabel', time_methods, 'XTickLabelRotation', 45);
    title('Execution Time Comparison');
    ylabel('Time (seconds)');
    grid on;
end

fprintf('✓ Example completed!\n\n');
