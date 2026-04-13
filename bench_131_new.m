% Benchmark newer NNLS algorithms on 131x131 Tikhonov
% FAST-NNLS, PG-BB, SI-NNLS  (OMP FP64/FP32, CUDA FP64/FP32)
% Also includes Classic AS for reference.

diary('/tmp/bench_131_new.log');
diary on;

proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);
src_root = fullfile(proj, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

% Warm-up GPU
try
    g = gpuDevice;
    fprintf('GPU: %s\n', g.Name);
    tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;
catch
    fprintf('GPU not available\n');
end

lambda = 0.001;
num_threads = 8;
base = '/home/matlab/ceshi1';

n = 131;
fprintf('\nGrid %dx%d\n', n, n);
folder = fullfile(base, sprintf('ceshi%d', n));
tic;
A = csvread(fullfile(folder, 'System_Matrix_3D.csv'));
v = flipud(csvread(fullfile(folder, sprintf('PSF%d.csv', n)), 1, 2));
fprintf('Load: %.1f s\n', toc);

[~, nA] = size(A);
B = [A; sqrt(lambda) * eye(nA)];
d = [v; zeros(nA, 1)];
clear A;
fprintf('B: %dx%d\n', size(B,1), size(B,2));

fprintf('\n[baseline] lsqnonneg ... '); tic;
x_ref = lsqnonneg(B, d);
t_ml = toc;
fprintf('%.3f s\n', t_ml);

solvers = {
    % Classic AS (baseline reference)
    'AS    OMP  FP64',  @() nnls_active_set_fp64_omp(B, d, num_threads)
    'AS    OMP  FP32',  @() nnls_active_set_fp32_omp(B, d, num_threads)
    'AS    CUDA FP64',  @() nnls_active_set_fp64_cuda(B, d)
    'AS    CUDA FP32',  @() nnls_active_set_fp32_cuda(B, d)
    % FAST-NNLS
    'FAST  OMP  FP64',  @() nnls_fast_nnls_fp64_omp(B, d, num_threads)
    'FAST  OMP  FP32',  @() nnls_fast_nnls_fp32_omp(B, d, num_threads)
    'FAST  CUDA FP64',  @() nnls_fast_nnls_fp64_cuda(B, d)
    'FAST  CUDA FP32',  @() nnls_fast_nnls_fp32_cuda(B, d)
    % PG-BB
    'PGBB  OMP  FP64',  @() nnls_pgbb_fp64_omp(B, d, num_threads)
    'PGBB  OMP  FP32',  @() nnls_pgbb_fp32_omp(B, d, num_threads)
    'PGBB  CUDA FP64',  @() nnls_pgbb_fp64_cuda(B, d)
    'PGBB  CUDA FP32',  @() nnls_pgbb_fp32_cuda(B, d)
    % SI-NNLS
    'SI    OMP  FP64',  @() nnls_si_nnls_fp64_omp(B, d, num_threads)
    'SI    OMP  FP32',  @() nnls_si_nnls_fp32_omp(B, d, num_threads)
    'SI    CUDA FP64',  @() nnls_si_nnls_fp64_cuda(B, d)
    'SI    CUDA FP32',  @() nnls_si_nnls_fp32_cuda(B, d)
};

fprintf('\n  %-18s %10s %10s %10s %s\n', 'Solver', 'Time(s)', 'Speedup', 'relErr', 'Status');
fprintf('  %s\n', repmat('-', 1, 64));

for k = 1:size(solvers, 1)
    name = solvers{k,1}; func = solvers{k,2};
    try
        tic; x = func(); t = toc;
        x = double(x(:));
        rel_err = norm(x - x_ref) / max(norm(x_ref), 1e-12);
        res = norm(B*x - d);
        ref_norm = norm(d);
        nn = all(x >= -1e-6);
        status = 'OK';
        if ~nn, status = 'FAIL(nn)'; end
        if res > ref_norm*10, status = 'FAIL(res)'; end
        fprintf('  %-18s %10.3f %9.1fx %10.2e %s\n', ...
                name, t, t_ml/t, rel_err, status);
    catch ME
        fprintf('  %-18s %10s %10s %10s ERR: %s\n', ...
                name, '--', '--', '--', ME.message);
    end
end

diary off;
exit;
