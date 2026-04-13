%% Quick CUDA benchmark (fast solvers only - skip slow active-set)
clear; clc;
src_root = fullfile(fileparts(mfilename('fullpath')), '..', 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

m = 2000; n = 1000; rng(42);
C = randn(m, n); x_true = abs(randn(n, 1));
d = C * x_true + 0.01 * randn(m, 1);

g = gpuDevice;
tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;

fprintf('=== CUDA Benchmark (2000x1000) on %s ===\n', g.Name);
fprintf('%-30s %10s %12s %6s\n', 'Solver', 'Time(s)', 'Residual', 'nn?');
fprintf('%s\n', repmat('-', 1, 62));

solvers = {
    '19 PG-BB FP64 CUDA',      @(C,d) nnls_pgbb_fp64_cuda(C, d)
    '20 PG-BB FP32 CUDA',      @(C,d) nnls_pgbb_fp32_cuda(C, d)
    '15 FNNLS FP64 CUDA',      @(C,d) nnls_fnnls_fp64_cuda(C, d)
    '16 FNNLS FP32 CUDA',      @(C,d) nnls_fnnls_fp32_cuda(C, d)
    '07 Classic GP FP64 CUDA', @(C,d) nnls_gradient_projection_fp64_cuda(C, d)
    '08 Classic GP FP32 CUDA', @(C,d) nnls_gradient_projection_fp32_cuda(C, d)
};

for k = 1:size(solvers, 1)
    name = solvers{k, 1};
    func = solvers{k, 2};
    fprintf('%-30s ', name);
    try
        tic; x = func(C, d); t = toc;
        x = double(x(:));
        res = norm(C * x - d);
        nn = all(x >= -1e-6);
        nn_str = 'YES'; if ~nn, nn_str = 'NO'; end
        fprintf('%10.4f %12.4e %6s\n', t, res, nn_str);
    catch ME
        fprintf('%10s %12s %6s  FAIL: %s\n', 'ERR', 'N/A', 'N/A', ME.message);
    end
end

fprintf('\nDone.\n');
