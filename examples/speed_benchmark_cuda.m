%% Speed Benchmark — CUDA solvers only (for standard vs unified memory comparison)
%
% Usage:
%   run('examples/speed_benchmark_cuda.m')

clear; clc;
fprintf('NNLS CUDA Speed Benchmark\n');
fprintf('=========================\n\n');

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
rng(42);

fprintf('Problem: %d x %d (%.1f MB FP64)\n', m, n, m*n*8/1e6);
C = randn(m, n);
x_true = abs(randn(n, 1));
d = C * x_true + 0.01 * randn(m, 1);
d_norm = norm(d);

% GPU info
try
    g = gpuDevice;
    fprintf('GPU: %s (%.0f MB)\n\n', g.Name, g.TotalMemory/1e6);
    tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;
catch
    fprintf('No GPU!\n\n');
    return;
end

%% CUDA solvers only (skip slow active-set)
solvers = {
    '19 PG-BB FP64 CUDA',        @(C,d) nnls_pgbb_fp64_cuda(C, d)
    '20 PG-BB FP32 CUDA',        @(C,d) nnls_pgbb_fp32_cuda(C, d)
    '15 FNNLS FP64 CUDA',        @(C,d) nnls_fnnls_fp64_cuda(C, d)
    '16 FNNLS FP32 CUDA',        @(C,d) nnls_fnnls_fp32_cuda(C, d)
    '07 Classic GP FP64 CUDA',   @(C,d) nnls_gradient_projection_fp64_cuda(C, d)
    '08 Classic GP FP32 CUDA',   @(C,d) nnls_gradient_projection_fp32_cuda(C, d)
    '03 Classic AS FP64 CUDA',   @(C,d) nnls_active_set_fp64_cuda(C, d)
    '04 Classic AS FP32 CUDA',   @(C,d) nnls_active_set_fp32_cuda(C, d)
};

fprintf('%-30s %10s %12s %6s  %s\n', 'Solver', 'Time(s)', 'Residual', 'nn?', 'Status');
fprintf('%s\n', repmat('-', 1, 72));

for k = 1:size(solvers, 1)
    name = solvers{k, 1};
    func = solvers{k, 2};
    fprintf('%-30s ', name);
    try
        tic;
        x = func(C, d);
        t = toc;

        x = double(x(:));
        res = norm(C * x - d);
        nn = all(x >= -1e-6);

        status = 'OK';
        if ~nn, status = 'FAIL(nn)'; end
        if res > d_norm, status = 'FAIL(res)'; end

        nn_str = 'YES'; if ~nn, nn_str = 'NO'; end
        fprintf('%10.4f %12.4e %6s  %s\n', t, res, nn_str, status);
    catch ME
        fprintf('%10s %12s %6s  FAIL: %s\n', 'ERR', 'N/A', 'N/A', ME.message);
    end
end

fprintf('\nDone.\n');
