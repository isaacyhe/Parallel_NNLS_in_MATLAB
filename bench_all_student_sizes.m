% Full benchmark: ALL 5 student paper sizes with student's actual data
% Uses Tikhonov regularization: B = [A; sqrt(lambda)*I]
% Pre-generated system matrices from ceshi1/

src_root = fullfile(pwd, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

% GPU warm-up
g = gpuDevice;
fprintf('GPU: %s (%.0f MB)\n', g.Name, g.TotalMemory/1e6);
tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;

lambda = 0.001;
num_threads = 8;

%% All 5 sizes from ceshi1
sizes = [21, 81, 111, 121, 131];
base = '/home/matlab/ceshi1';

results = zeros(length(sizes), 5);  % matlab, omp, cu64, cu32, fast_cu64

for si = 1:length(sizes)
    n = sizes(si);
    folder = fullfile(base, sprintf('ceshi%d', n));
    sm_csv = fullfile(folder, 'System_Matrix_3D.csv');
    psf_csv = fullfile(folder, sprintf('PSF%d.csv', n));

    fprintf('\n================================================================\n');
    fprintf('Grid %dx%d\n', n, n);
    fprintf('================================================================\n');

    % Load system matrix
    fprintf('Loading System_Matrix_3D.csv ...');
    tic; A = csvread(sm_csv); t_load = toc;
    [mA, nA] = size(A);
    fprintf(' %dx%d in %.1f s\n', mA, nA, t_load);

    % Load measurement
    v = flipud(csvread(psf_csv, 1, 2));

    % Tikhonov regularization
    B = [A; sqrt(lambda) * eye(nA)];
    d = [v; zeros(nA, 1)];
    [mB, nB] = size(B);
    fprintf('Augmented: %dx%d  (%.1f GB FP64)\n', mB, nB, mB*nB*8/1e9);
    clear A;

    % 1) MATLAB lsqnonneg
    fprintf('  MATLAB lsqnonneg ... ');
    tic; x1 = lsqnonneg(B, d); t_ml = toc;
    fprintf('%.3f s\n', t_ml);
    results(si,1) = t_ml;

    % 2) Our OMP FP64 (QR-based)
    fprintf('  Classic AS FP64 OMP (QR, %d thr) ... ', num_threads);
    try
        tic; x2 = nnls_active_set_fp64_omp(B, d, num_threads); t_omp = toc;
        fprintf('%.3f s  (%.1fx)\n', t_omp, t_ml/t_omp);
    catch ME
        fprintf('FAIL: %s\n', ME.message); t_omp = NaN;
    end
    results(si,2) = t_omp;

    % 3) CUDA FP64
    fprintf('  Classic AS FP64 CUDA ... ');
    try
        tic; x3 = nnls_active_set_fp64_cuda(B, d); t_cu64 = toc;
        fprintf('%.3f s  (%.1fx)\n', t_cu64, t_ml/t_cu64);
    catch ME
        fprintf('FAIL: %s\n', ME.message); t_cu64 = NaN;
    end
    results(si,3) = t_cu64;

    % 4) CUDA FP32
    fprintf('  Classic AS FP32 CUDA ... ');
    try
        tic; x4 = nnls_active_set_fp32_cuda(B, d); t_cu32 = toc;
        fprintf('%.3f s  (%.1fx)\n', t_cu32, t_ml/t_cu32);
    catch ME
        fprintf('FAIL: %s\n', ME.message); t_cu32 = NaN;
    end
    results(si,4) = t_cu32;

    % 5) FAST-NNLS FP64 CUDA (our improved solver)
    fprintf('  FAST-NNLS FP64 CUDA ... ');
    try
        tic; x5 = nnls_fast_nnls_fp64_cuda(B, d); t_fn64 = toc;
        fprintf('%.3f s  (%.1fx)\n', t_fn64, t_ml/t_fn64);
    catch ME
        fprintf('FAIL: %s\n', ME.message); t_fn64 = NaN;
    end
    results(si,5) = t_fn64;

    clear B d;
end

%% Summary
fprintf('\n\n');
fprintf('================================================================\n');
fprintf('SUMMARY — Tesla V100-SXM2-16GB (FP32:FP64 = 2:1)\n');
fprintf('================================================================\n\n');

fprintf('%-8s %8s | %10s %10s %10s %10s %10s\n', ...
    'Grid', 'Sys NxN', 'MATLAB(s)', 'OMP8(s)', 'CU-64(s)', 'CU-32(s)', 'FNNLS-64');
fprintf('%s\n', repmat('-', 1, 80));
for si = 1:length(sizes)
    n = sizes(si);
    N = n*n;
    fprintf('%-8s %8d | ', sprintf('%dx%d',n,n), N);
    for col = 1:5
        if isnan(results(si,col))
            fprintf('%10s ', 'FAIL');
        else
            fprintf('%10.3f ', results(si,col));
        end
    end
    fprintf('\n');
end

fprintf('\nSpeedups vs MATLAB lsqnonneg:\n');
fprintf('%-8s | %10s %10s %10s %10s\n', 'Grid', 'OMP8', 'CUDA-64', 'CUDA-32', 'FNNLS-64');
fprintf('%s\n', repmat('-', 1, 55));
for si = 1:length(sizes)
    n = sizes(si);
    fprintf('%-8s | ', sprintf('%dx%d',n,n));
    for col = 2:5
        if isnan(results(si,col))
            fprintf('%10s ', 'FAIL');
        else
            fprintf('%9.1fx ', results(si,1)/results(si,col));
        end
    end
    fprintf('\n');
end

fprintf('\nStudent paper (RTX 4000 Ada, FP32:FP64=32:1):\n');
fprintf('  OMP 8T:   [5.1, 24.7, 28.2, 32.9, 25.0]\n');
fprintf('  CUDA-64:  [2.1, 20.1, 23.1, 27.6, 20.5]\n');
fprintf('  CUDA-32:  [5.7, 159,  736,  837,  995]\n');
fprintf('\nDone.\n');
exit;
