% Quick benchmark: Cholesky-based #11/12 vs MATLAB lsqnonneg on
% student's Tikhonov-regularized MPI problem (size 111 only)

diary('/tmp/bench_chol.log');
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

lambda = 0.001;
num_threads = 8;
base = '/home/matlab/ceshi1';

n = 111;
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

fprintf('\nlsqnonneg ... '); tic;
x_ref = lsqnonneg(B, d);
t_ml = toc;
fprintf('%.3f s\n', t_ml);

fprintf('Classic AS FP64 OMP (new) ... '); tic;
x1 = nnls_active_set_fp64_omp(B, d, num_threads);
t_omp64 = toc;
rel1 = norm(x1 - x_ref) / norm(x_ref);
fprintf('%.3f s  (%.1fx)  relErr=%.2e\n', t_omp64, t_ml/t_omp64, rel1);

fprintf('Classic AS FP32 OMP (new) ... '); tic;
x2 = nnls_active_set_fp32_omp(B, d, num_threads);
t_omp32 = toc;
rel2 = norm(x2 - x_ref) / norm(x_ref);
fprintf('%.3f s  (%.1fx)  relErr=%.2e\n', t_omp32, t_ml/t_omp32, rel2);

diary off;
exit;
