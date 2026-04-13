%% Basic Usage Example for All 30 NNLS Solvers
% This script demonstrates and benchmarks all NNLS implementations
%
% Prerequisites:
%   - Build all MEX files using build/build_all.m
%   - Run this script from the examples/ directory or project root
%
% License: MIT

%% Setup
clear; clc;
fprintf('NNLS Solver Benchmark — All 30 Implementations\n');
fprintf('================================================\n\n');

% Add all source directories to path
src_root = fullfile(fileparts(mfilename('fullpath')), '..', 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

%% Generate Test Problem
fprintf('Generating test problem...\n');

m = 2000;  % rows
n = 1000;  % columns
num_threads = 8;

rng(42);
C = randn(m, n);
x_true = abs(randn(n, 1));
d = C * x_true + 0.01 * randn(m, 1);

fprintf('  Problem size: C is %dx%d, d is %dx1\n', m, n, m);
fprintf('  Threads for OpenMP/MATLAB-MT: %d\n\n', num_threads);

% Non-negative data for SI-NNLS solvers (requires C >= 0, d >= 0)
C_nn = abs(C);
d_nn = C_nn * x_true + 0.01 * abs(randn(m, 1));

%% Define all 30 solvers
% Format: {name, function_handle, arg_style}
%   arg_style: 'omp' = (C,d,threads), 'cuda' = (C,d), 'st' = (C,d), 'mt' = (C,d,threads)
%   '_nn' suffix = uses non-negative C_nn, d_nn instead

solvers = {
    % MATLAB (old active-set)
    '01 MATLAB lsqnonneg ST',    @nnls_matlab_lsqnonneg_st,     'st'
    '02 MATLAB lsqnonneg MT',    @nnls_matlab_lsqnonneg_mt,     'mt'
    % MATLAB (old gradient projection)
    '03 MATLAB Classic GP ST',   @nnls_classic_gp_st,            'st'
    '04 MATLAB Classic GP MT',   @nnls_classic_gp_mt,            'mt'
    % MATLAB (newest active-set)
    '05 MATLAB FAST-NNLS ST',   @nnls_fast_nnls_st,             'st'
    '06 MATLAB FAST-NNLS MT',   @nnls_fast_nnls_mt,             'mt'
    % MATLAB (PG-BB)
    '07 MATLAB PG-BB ST',       @nnls_pgbb_st,                  'st'
    '08 MATLAB PG-BB MT',       @nnls_pgbb_mt,                  'mt'
    % MATLAB (newest gradient projection)
    '09 MATLAB SI-NNLS ST',     @nnls_si_nnls_st,               'st_nn'
    '10 MATLAB SI-NNLS MT',     @nnls_si_nnls_mt,               'mt_nn'
    % OMP (old active-set)
    '11 Classic AS FP64 OMP',    @nnls_active_set_fp64_omp,      'omp'
    '12 Classic AS FP32 OMP',    @nnls_active_set_fp32_omp,      'omp'
    % OMP (old gradient projection)
    '13 Classic GP FP64 OMP',    @nnls_gradient_projection_fp64_omp,  'omp'
    '14 Classic GP FP32 OMP',    @nnls_gradient_projection_fp32_omp,  'omp'
    % OMP (newest active-set)
    '15 FAST-NNLS FP64 OMP',    @nnls_fast_nnls_fp64_omp,       'omp'
    '16 FAST-NNLS FP32 OMP',    @nnls_fast_nnls_fp32_omp,       'omp'
    % OMP (PG-BB)
    '17 PG-BB FP64 OMP',        @nnls_pgbb_fp64_omp,            'omp'
    '18 PG-BB FP32 OMP',        @nnls_pgbb_fp32_omp,            'omp'
    % OMP (newest gradient projection)
    '19 SI-NNLS FP64 OMP',      @nnls_si_nnls_fp64_omp,         'omp_nn'
    '20 SI-NNLS FP32 OMP',      @nnls_si_nnls_fp32_omp,         'omp_nn'
    % CUDA (old active-set)
    '21 Classic AS FP64 CUDA',   @nnls_active_set_fp64_cuda,     'cuda'
    '22 Classic AS FP32 CUDA',   @nnls_active_set_fp32_cuda,     'cuda'
    % CUDA (old gradient projection)
    '23 Classic GP FP64 CUDA',   @nnls_gradient_projection_fp64_cuda, 'cuda'
    '24 Classic GP FP32 CUDA',   @nnls_gradient_projection_fp32_cuda, 'cuda'
    % CUDA (newest active-set)
    '25 FAST-NNLS FP64 CUDA',   @nnls_fast_nnls_fp64_cuda,      'cuda'
    '26 FAST-NNLS FP32 CUDA',   @nnls_fast_nnls_fp32_cuda,      'cuda'
    % CUDA (PG-BB)
    '27 PG-BB FP64 CUDA',       @nnls_pgbb_fp64_cuda,           'cuda'
    '28 PG-BB FP32 CUDA',       @nnls_pgbb_fp32_cuda,           'cuda'
    % CUDA (newest gradient projection)
    '29 SI-NNLS FP64 CUDA',     @nnls_si_nnls_fp64_cuda,        'cuda_nn'
    '30 SI-NNLS FP32 CUDA',     @nnls_si_nnls_fp32_cuda,        'cuda_nn'
};

%% Run all solvers
num_solvers = size(solvers, 1);
results = struct('name', {}, 'time', {}, 'residual', {}, 'nonneg', {}, 'nnz_count', {}, 'success', {});

fprintf('=== Running %d solvers ===\n\n', num_solvers);

for k = 1:num_solvers
    name = solvers{k, 1};
    func = solvers{k, 2};
    style = solvers{k, 3};

    fprintf('%-30s ', name);

    try
        tic;
        switch style
            case 'omp'
                x = func(C, d, num_threads);
            case 'cuda'
                x = func(C, d);
            case 'st'
                x = func(C, d);
            case 'mt'
                x = func(C, d, num_threads);
            case 'omp_nn'
                x = func(C_nn, d_nn, num_threads);
            case 'cuda_nn'
                x = func(C_nn, d_nn);
            case 'st_nn'
                x = func(C_nn, d_nn);
            case 'mt_nn'
                x = func(C_nn, d_nn, num_threads);
        end
        elapsed = toc;

        x = double(x(:));
        if contains(style, '_nn')
            res = norm(C_nn * x - d_nn);
        else
            res = norm(C * x - d);
        end
        nn = all(x >= -1e-6);
        nnz_c = sum(abs(x) > 1e-10);

        results(end+1).name = name; %#ok<SAGROW>
        results(end).time = elapsed;
        results(end).residual = res;
        results(end).nonneg = nn;
        results(end).nnz_count = nnz_c;
        results(end).success = true;

        fprintf('%.4fs  res=%.2e  nnz=%d  nn=%d\n', elapsed, res, nnz_c, nn);
    catch ME
        results(end+1).name = name; %#ok<SAGROW>
        results(end).time = NaN;
        results(end).residual = NaN;
        results(end).nonneg = false;
        results(end).nnz_count = 0;
        results(end).success = false;

        fprintf('FAILED: %s\n', ME.message);
    end
end

%% Summary Table
fprintf('\n================================================\n');
fprintf('Summary\n');
fprintf('================================================\n\n');

fprintf('%-30s %10s %12s %6s\n', 'Solver', 'Time (s)', 'Residual', 'OK?');
fprintf('%s\n', repmat('-', 1, 62));

for k = 1:length(results)
    if results(k).success
        fprintf('%-30s %10.4f %12.2e %6s\n', ...
            results(k).name, results(k).time, results(k).residual, ...
            tf_str(results(k).nonneg));
    else
        fprintf('%-30s %10s %12s %6s\n', results(k).name, 'FAIL', 'N/A', 'N/A');
    end
end

successful = [results.success];
fprintf('\nSuccessful: %d/%d\n', sum(successful), length(successful));

%% Visualization
fprintf('\nGenerating comparison plots...\n');

ok = find([results.success]);
if length(ok) >= 2
    figure('Position', [100, 100, 1400, 500]);

    % Plot 1: Execution times
    subplot(1, 3, 1);
    times = [results(ok).time];
    barh(times);
    set(gca, 'YTick', 1:length(ok), 'YTickLabel', {results(ok).name}, 'FontSize', 7);
    xlabel('Time (seconds)');
    title('Execution Time');
    grid on;

    % Plot 2: Residuals
    subplot(1, 3, 2);
    residuals = [results(ok).residual];
    barh(residuals);
    set(gca, 'YTick', 1:length(ok), 'YTickLabel', {results(ok).name}, 'FontSize', 7);
    xlabel('||Cx - d||');
    title('Residual');
    grid on;

    % Plot 3: Speedup relative to slowest
    subplot(1, 3, 3);
    max_time = max(times);
    speedup = max_time ./ times;
    barh(speedup);
    set(gca, 'YTick', 1:length(ok), 'YTickLabel', {results(ok).name}, 'FontSize', 7);
    xlabel('Speedup (x)');
    title('Relative Speedup');
    grid on;
end

fprintf('Done!\n\n');

%% Helper
function s = tf_str(val)
    if val
        s = 'YES';
    else
        s = 'NO';
    end
end
