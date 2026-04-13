% Benchmark matching the student's EXACT conditions
% Uses the student's pre-generated System_Matrix_3D.csv files
% with Tikhonov regularization: B = [A; sqrt(lambda)*I], d = [v; 0]
%
% Student's sizes refer to GRID resolution, not matrix size:
%   21x21 grid  -> 441x441 system matrix   -> 882x441 augmented
%   81x81 grid  -> 6561x6561 system matrix  -> 13122x6561 augmented
%   111x111 grid -> 12321x12321 system matrix -> 24642x12321 augmented

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

%% Load student's actual data
student_data = {
    '/home/matlab/ceshi2/ceshi21',  'PSF21.csv',  21
    '/home/matlab/ceshi2/ceshi81',  'PSF81.csv',  81
    '/home/matlab/ceshi2/ceshi111', 'PSF111.csv', 111
};

for si = 1:size(student_data, 1)
    folder = student_data{si, 1};
    psf_file = student_data{si, 2};
    grid_n = student_data{si, 3};

    fprintf('\n================================================================\n');
    fprintf('Grid: %dx%d\n', grid_n, grid_n);
    fprintf('================================================================\n');

    % Load system matrix
    sm_path = fullfile(folder, 'System_Matrix_3D.csv');
    if ~exist(sm_path, 'file')
        fprintf('  System_Matrix_3D.csv not found, skipping\n');
        continue;
    end
    fprintf('Loading %s ...\n', sm_path);
    tic; A = csvread(sm_path); t_load = toc;
    [m_A, n_A] = size(A);
    fprintf('  A: %d x %d (loaded in %.1f s)\n', m_A, n_A, t_load);

    % Load PSF as measurement vector (student's approach)
    psf_path = fullfile(folder, psf_file);
    fprintf('Loading %s ...\n', psf_path);
    v = flipud(csvread(psf_path, 1, 2));
    fprintf('  v: %d x 1\n', length(v));

    % Tikhonov regularization (exactly as student did)
    B = [A; sqrt(lambda) * eye(n_A)];
    d = [v; zeros(n_A, 1)];
    [m_B, n_B] = size(B);
    fprintf('  Augmented B: %d x %d (Tikhonov lambda=%.4f)\n', m_B, n_B, lambda);
    fprintf('  B memory: %.1f MB (FP64)\n', m_B * n_B * 8 / 1e6);

    % ============================================================
    % Benchmark: MATLAB baseline (lsqnonneg) — what the student used
    % ============================================================
    fprintf('\nRunning MATLAB lsqnonneg (baseline)...\n');
    tic; x_ref = lsqnonneg(B, d); t_matlab = toc;
    fprintf('  MATLAB lsqnonneg: %.3f s\n', t_matlab);

    % ============================================================
    % OMP FP64 (Classic Active-Set, 8 threads)
    % ============================================================
    fprintf('Running Classic AS FP64 OMP (%d threads)...\n', num_threads);
    try
        tic; x_omp = nnls_active_set_fp64_omp(B, d, num_threads); t_omp = toc;
        fprintf('  OMP FP64: %.3f s (speedup: %.1fx)\n', t_omp, t_matlab/t_omp);
    catch ME
        fprintf('  OMP FP64: FAILED - %s\n', ME.message);
        t_omp = NaN;
    end

    % ============================================================
    % CUDA FP64 (Classic Active-Set)
    % ============================================================
    fprintf('Running Classic AS FP64 CUDA...\n');
    try
        tic; x_cu64 = nnls_active_set_fp64_cuda(B, d); t_cu64 = toc;
        fprintf('  CUDA FP64: %.3f s (speedup: %.1fx)\n', t_cu64, t_matlab/t_cu64);
    catch ME
        fprintf('  CUDA FP64: FAILED - %s\n', ME.message);
        t_cu64 = NaN;
    end

    % ============================================================
    % CUDA FP32 (Classic Active-Set)
    % ============================================================
    fprintf('Running Classic AS FP32 CUDA...\n');
    try
        tic; x_cu32 = nnls_active_set_fp32_cuda(B, d); t_cu32 = toc;
        fprintf('  CUDA FP32: %.3f s (speedup: %.1fx)\n', t_cu32, t_matlab/t_cu32);
    catch ME
        fprintf('  CUDA FP32: FAILED - %s\n', ME.message);
        t_cu32 = NaN;
    end

    % ============================================================
    % Summary table
    % ============================================================
    fprintf('\n--- Summary: Grid %dx%d (Matrix %dx%d, Augmented %dx%d) ---\n', ...
        grid_n, grid_n, m_A, n_A, m_B, n_B);
    fprintf('%-25s %12s %10s\n', 'Solver', 'Time(s)', 'Speedup');
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf('%-25s %12.3f %10s\n', 'MATLAB lsqnonneg', t_matlab, '1.0x');
    if ~isnan(t_omp)
        fprintf('%-25s %12.3f %10.1fx\n', 'OMP FP64 (8 threads)', t_omp, t_matlab/t_omp);
    end
    if ~isnan(t_cu64)
        fprintf('%-25s %12.3f %10.1fx\n', 'CUDA FP64', t_cu64, t_matlab/t_cu64);
    end
    if ~isnan(t_cu32)
        fprintf('%-25s %12.3f %10.1fx\n', 'CUDA FP32', t_cu32, t_matlab/t_cu32);
    end

    clear A B d v;
end

fprintf('\nDone.\n');
exit;
