% Benchmark FAST-NNLS CUDA (GPU-resident rewrite) on 131 Tikhonov.
% Uses cached x_ref from /tmp/x_ref_131.mat.

diary('/tmp/bench_131_fast_cuda.log'); diary on;

proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);
src_root = fullfile(proj, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

try
    g = gpuDevice;
    fprintf('GPU: %s\n', g.Name);
    tmp = gpuArray(randn(500)); tmp*tmp'; clear tmp;
catch
end

lambda = 0.001;
base = '/home/matlab/ceshi1';
n = 131;
folder = fullfile(base, sprintf('ceshi%d', n));
fprintf('Loading 131 ... '); tic;
A = csvread(fullfile(folder, 'System_Matrix_3D.csv'));
v = flipud(csvread(fullfile(folder, sprintf('PSF%d.csv', n)), 1, 2));
fprintf('%.1f s\n', toc);
[~, nA] = size(A);
B = [A; sqrt(lambda) * eye(nA)];
d = [v; zeros(nA, 1)];
clear A;
fprintf('B: %dx%d\n', size(B,1), size(B,2));

ref_file = '/tmp/x_ref_131.mat';
S = load(ref_file); x_ref = S.x_ref; t_ml = S.t_ml;
fprintf('x_ref cached (lsqnonneg = %.3f s)\n', t_ml);

solvers = {
    'FAST  CUDA FP64',  @() nnls_fast_nnls_fp64_cuda(B, d)
    'FAST  CUDA FP32',  @() nnls_fast_nnls_fp32_cuda(B, d)
};

fprintf('\n  %-18s %10s %10s %10s %s\n', 'Solver', 'Time(s)', 'Speedup', 'relErr', 'Status');
fprintf('  %s\n', repmat('-', 1, 64));

for k = 1:size(solvers, 1)
    name = solvers{k,1}; func = solvers{k,2};
    diary off; diary on;
    try
        tic; x = func(); t = toc;
        x = double(x(:));
        rel_err = norm(x - x_ref) / max(norm(x_ref), 1e-12);
        nn = all(x >= -1e-6);
        status = 'OK'; if ~nn, status = 'FAIL(nn)'; end
        fprintf('  %-18s %10.3f %9.1fx %10.2e %s\n', ...
                name, t, t_ml/t, rel_err, status);
    catch ME
        fprintf('  %-18s %10s %10s %10s ERR: %s\n', name, '--', '--', '--', ME.message);
    end
    diary off; diary on;
end

diary off;
exit;
