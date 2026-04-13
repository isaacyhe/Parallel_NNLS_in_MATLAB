%% Speed Benchmark — Large Input
% Tests all 30 NNLS solvers one-by-one on a large problem
%
% Usage:
%   run('examples/speed_benchmark.m')
%   speed_benchmark           % from examples/ directory

clear; clc;
fprintf('NNLS Speed Benchmark — Large Input\n');
fprintf('===================================\n\n');

% Add source paths
src_root = fullfile(fileparts(mfilename('fullpath')), '..', 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

%% Problem setup
m = 2000; n = 1000;
num_threads = 8;
rng(42);

fprintf('Generating %d x %d test problem...\n', m, n);
tic;
C = randn(m, n);
x_true = abs(randn(n, 1));
d = C * x_true + 0.01 * randn(m, 1);
fprintf('  Generated in %.2f s (%.1f MB FP64)\n', toc, m*n*8/1e6);
d_norm = norm(d);
fprintf('  ||d|| = %.4e\n\n', d_norm);

% Non-negative data for SI-NNLS solvers
C_nn = abs(C);
d_nn = C_nn * x_true + 0.01 * abs(randn(m, 1));
d_nn_norm = norm(d_nn);

% GPU warm-up
try
    g = gpuDevice;
    fprintf('GPU: %s (%.0f MB HBM)\n', g.Name, g.TotalMemory/1e6);
    tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;
    fprintf('GPU warmed up.\n\n');
catch
    fprintf('No GPU available.\n\n');
end

%% Solver definitions
solvers = {
    % --- MATLAB (old active-set) ---
    '01 MATLAB lsqnonneg ST',    @(C,d) nnls_matlab_lsqnonneg_st(C, d)
    '02 MATLAB lsqnonneg MT',    @(C,d) nnls_matlab_lsqnonneg_mt(C, d, num_threads)
    % --- MATLAB (old gradient projection) ---
    '03 MATLAB Classic GP ST',   @(C,d) nnls_classic_gp_st(C, d)
    '04 MATLAB Classic GP MT',   @(C,d) nnls_classic_gp_mt(C, d, num_threads)
    % --- MATLAB (newest active-set) ---
    '05 MATLAB FAST-NNLS ST',   @(C,d) nnls_fast_nnls_st(C, d)
    '06 MATLAB FAST-NNLS MT',   @(C,d) nnls_fast_nnls_mt(C, d, num_threads)
    % --- MATLAB (PG-BB) ---
    '07 MATLAB PG-BB ST',       @(C,d) nnls_pgbb_st(C, d)
    '08 MATLAB PG-BB MT',       @(C,d) nnls_pgbb_mt(C, d, num_threads)
    % --- MATLAB (newest gradient projection) ---
    '09 MATLAB SI-NNLS ST',     @(C,d) nnls_si_nnls_st(C_nn, d_nn)
    '10 MATLAB SI-NNLS MT',     @(C,d) nnls_si_nnls_mt(C_nn, d_nn, num_threads)
    % --- OMP (old active-set) ---
    '11 Classic AS FP64 OMP',    @(C,d) nnls_active_set_fp64_omp(C, d, num_threads)
    '12 Classic AS FP32 OMP',    @(C,d) nnls_active_set_fp32_omp(C, d, num_threads)
    % --- OMP (old gradient projection) ---
    '13 Classic GP FP64 OMP',    @(C,d) nnls_gradient_projection_fp64_omp(C, d, num_threads)
    '14 Classic GP FP32 OMP',    @(C,d) nnls_gradient_projection_fp32_omp(single(C), single(d), num_threads)
    % --- OMP (newest active-set) ---
    '15 FAST-NNLS FP64 OMP',    @(C,d) nnls_fast_nnls_fp64_omp(C, d, num_threads)
    '16 FAST-NNLS FP32 OMP',    @(C,d) nnls_fast_nnls_fp32_omp(C, d, num_threads)
    % --- OMP (PG-BB) ---
    '17 PG-BB FP64 OMP',        @(C,d) nnls_pgbb_fp64_omp(C, d, num_threads)
    '18 PG-BB FP32 OMP',        @(C,d) nnls_pgbb_fp32_omp(C, d, num_threads)
    % --- OMP (newest gradient projection) ---
    '19 SI-NNLS FP64 OMP',      @(C,d) nnls_si_nnls_fp64_omp(C_nn, d_nn, num_threads)
    '20 SI-NNLS FP32 OMP',      @(C,d) nnls_si_nnls_fp32_omp(C_nn, d_nn, num_threads)
    % --- CUDA (old active-set) ---
    '21 Classic AS FP64 CUDA',   @(C,d) nnls_active_set_fp64_cuda(C, d)
    '22 Classic AS FP32 CUDA',   @(C,d) nnls_active_set_fp32_cuda(C, d)
    % --- CUDA (old gradient projection) ---
    '23 Classic GP FP64 CUDA',   @(C,d) nnls_gradient_projection_fp64_cuda(C, d)
    '24 Classic GP FP32 CUDA',   @(C,d) nnls_gradient_projection_fp32_cuda(C, d)
    % --- CUDA (newest active-set) ---
    '25 FAST-NNLS FP64 CUDA',   @(C,d) nnls_fast_nnls_fp64_cuda(C, d)
    '26 FAST-NNLS FP32 CUDA',   @(C,d) nnls_fast_nnls_fp32_cuda(C, d)
    % --- CUDA (PG-BB) ---
    '27 PG-BB FP64 CUDA',       @(C,d) nnls_pgbb_fp64_cuda(C, d)
    '28 PG-BB FP32 CUDA',       @(C,d) nnls_pgbb_fp32_cuda(C, d)
    % --- CUDA (newest gradient projection) ---
    '29 SI-NNLS FP64 CUDA',     @(C,d) nnls_si_nnls_fp64_cuda(C_nn, d_nn)
    '30 SI-NNLS FP32 CUDA',     @(C,d) nnls_si_nnls_fp32_cuda(C_nn, d_nn)
};

%% Run solvers one by one
fprintf('%-30s %10s %12s %6s  %s\n', 'Solver', 'Time(s)', 'Residual', 'nn?', 'Status');
fprintf('%s\n', repmat('-', 1, 72));

names_out = {};
times = [];
residuals = [];

for k = 1:size(solvers, 1)
    name = solvers{k, 1};
    func = solvers{k, 2};
    fprintf('%-30s ', name);
    try
        tic;
        x = func(C, d);
        t = toc;

        x = double(x(:));
        if contains(name, 'SI-NNLS')
            res = norm(C_nn * x - d_nn);
            nn = all(x >= -1e-6);
            status = 'OK';
            if ~nn, status = 'FAIL(nn)'; end
            if res > d_nn_norm, status = 'FAIL(res)'; end
        else
            res = norm(C * x - d);
            nn = all(x >= -1e-6);
            status = 'OK';
            if ~nn, status = 'FAIL(nn)'; end
            if res > d_norm, status = 'FAIL(res)'; end
        end

        nn_str = 'YES'; if ~nn, nn_str = 'NO'; end
        fprintf('%10.4f %12.4e %6s  %s\n', t, res, nn_str, status);
        names_out{end+1} = name; %#ok<SAGROW>
        times(end+1) = t; %#ok<SAGROW>
        residuals(end+1) = res; %#ok<SAGROW>
    catch ME
        fprintf('%10s %12s %6s  FAIL: %s\n', 'ERR', 'N/A', 'N/A', ME.message);
    end
end

%% Summary
fprintf('\n===================================\n');
fprintf('Problem: %d x %d, threads=%d\n', m, n, num_threads);
fprintf('===================================\n');

if ~isempty(times)
    [~, fastest_idx] = min(times);
    fprintf('Fastest: %s (%.4f s)\n', names_out{fastest_idx}, times(fastest_idx));
    [~, slowest_idx] = max(times);
    fprintf('Slowest: %s (%.4f s)\n', names_out{slowest_idx}, times(slowest_idx));
    fprintf('Speedup (slowest/fastest): %.1fx\n', times(slowest_idx) / times(fastest_idx));
end
fprintf('\nDone.\n');
