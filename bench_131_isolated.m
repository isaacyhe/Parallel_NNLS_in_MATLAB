% Run ONE new NNLS solver on 131, isolated so a crash doesn't mask others.
% Pre-cache x_ref so lsqnonneg only runs once.
% Usage: matlab -batch "TARGET=N; run('bench_131_isolated.m')"

proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);
src_root = fullfile(proj, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

diary(sprintf('/tmp/bench_iso_%d.log', TARGET));
diary on;

lambda = 0.001;
num_threads = 8;
base = '/home/matlab/ceshi1';

n = 131;
folder = fullfile(base, sprintf('ceshi%d', n));
fprintf('[%s] Loading 131 ...\n', datestr(now,'HH:MM:SS'));
A = csvread(fullfile(folder, 'System_Matrix_3D.csv'));
v = flipud(csvread(fullfile(folder, sprintf('PSF%d.csv', n)), 1, 2));
[~, nA] = size(A);
B = [A; sqrt(lambda) * eye(nA)];
d = [v; zeros(nA, 1)];
clear A;
fprintf('B: %dx%d\n', size(B,1), size(B,2));

ref_file = '/tmp/x_ref_131.mat';
if exist(ref_file, 'file')
    S = load(ref_file); x_ref = S.x_ref; t_ml = S.t_ml;
    fprintf('x_ref cached (lsqnonneg = %.3f s)\n', t_ml);
else
    fprintf('Computing x_ref via lsqnonneg ...\n');
    tic; x_ref = lsqnonneg(B, d); t_ml = toc;
    save(ref_file, 'x_ref', 't_ml', '-v7.3');
    fprintf('lsqnonneg = %.3f s\n', t_ml);
end

solvers = {
    1, 'FAST OMP  FP64', @() nnls_fast_nnls_fp64_omp(B, d, num_threads);
    2, 'FAST OMP  FP32', @() nnls_fast_nnls_fp32_omp(B, d, num_threads);
    3, 'FAST CUDA FP64', @() nnls_fast_nnls_fp64_cuda(B, d);
    4, 'FAST CUDA FP32', @() nnls_fast_nnls_fp32_cuda(B, d);
    5, 'PGBB OMP  FP64', @() nnls_pgbb_fp64_omp(B, d, num_threads);
    6, 'PGBB OMP  FP32', @() nnls_pgbb_fp32_omp(B, d, num_threads);
    7, 'PGBB CUDA FP64', @() nnls_pgbb_fp64_cuda(B, d);
    8, 'PGBB CUDA FP32', @() nnls_pgbb_fp32_cuda(B, d);
    9, 'SI   OMP  FP64', @() nnls_si_nnls_fp64_omp(B, d, num_threads);
   10, 'SI   OMP  FP32', @() nnls_si_nnls_fp32_omp(B, d, num_threads);
   11, 'SI   CUDA FP64', @() nnls_si_nnls_fp64_cuda(B, d);
   12, 'SI   CUDA FP32', @() nnls_si_nnls_fp32_cuda(B, d);
};

for s = 1:size(solvers, 1)
    if solvers{s,1} ~= TARGET, continue; end
    name = solvers{s,2}; func = solvers{s,3};
    fprintf('>>> [%s] Running %s ...\n', datestr(now,'HH:MM:SS'), name);
    diary off; diary on;
    try
        tic;
        x = func();
        t = toc;
        x = double(x(:));
        rel_err = norm(x - x_ref) / max(norm(x_ref), 1e-12);
        nn = all(x >= -1e-6);
        fprintf('<<< %s: %.3f s  (%.2fx)  relErr=%.2e  nn=%d\n', ...
                name, t, t_ml/t, rel_err, nn);
    catch ME
        fprintf('<<< %s: ERROR: %s\n', name, ME.message);
    end
end

diary off;
exit;
