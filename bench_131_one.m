% Run ONE of the 18 new-alg variants on 131, isolated so hangs are contained.
% Pre-caches B,d and x_ref in /tmp/bench_131_data.mat (first call builds it).
% Usage: matlab -batch "TARGET=N; run('bench_131_one.m')"

proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);
src_root = fullfile(proj, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

result_file = sprintf('/tmp/bench_131_result_%02d.txt', TARGET);
log_file    = sprintf('/tmp/bench_131_log_%02d.log',   TARGET);
diary(log_file); diary on;

try
    g = gpuDevice;
    fprintf('GPU: %s\n', g.Name);
    tmp = gpuArray(randn(500)); tmp*tmp'; clear tmp;
catch
end

data_cache = '/tmp/bench_131_data.mat';
ref_file   = '/tmp/x_ref_131.mat';

if exist(data_cache, 'file')
    fprintf('[%s] Loading cached B,d ... ', datestr(now,'HH:MM:SS'));
    tic; S = load(data_cache); B = S.B; d = S.d; fprintf('%.1f s\n', toc);
else
    lambda = 0.001;
    base = '/home/matlab/ceshi1';
    n = 131;
    folder = fullfile(base, sprintf('ceshi%d', n));
    fprintf('[%s] Loading 131 CSV ... ', datestr(now,'HH:MM:SS'));
    tic;
    A = csvread(fullfile(folder, 'System_Matrix_3D.csv'));
    v = flipud(csvread(fullfile(folder, sprintf('PSF%d.csv', n)), 1, 2));
    [~, nA] = size(A);
    B = [A; sqrt(lambda) * eye(nA)];
    d = [v; zeros(nA, 1)];
    clear A;
    fprintf('%.1f s\n', toc);
    save(data_cache, 'B', 'd', '-v7.3');
end
fprintf('B: %dx%d\n', size(B,1), size(B,2));

if exist(ref_file, 'file')
    S = load(ref_file); x_ref = S.x_ref; t_ml = S.t_ml;
    fprintf('x_ref cached (lsqnonneg = %.3f s)\n', t_ml);
else
    fprintf('Computing x_ref via lsqnonneg ... ');
    tic; x_ref = lsqnonneg(B, d); t_ml = toc;
    save(ref_file, 'x_ref', 't_ml', '-v7.3');
    fprintf('%.3f s\n', t_ml);
end

num_threads = 8;

% 30 variants, device-grouped: MATLAB (10) -> OMP (10) -> CUDA (10).
% Family order within each group: lsqnonneg/CAS, FAST, GP, ADMM, CG.
solvers = {
    % --- MATLAB (10) ---
     1, 'LSQNN MATLAB ST',  @() nnls_matlab_lsqnonneg_st(B, d)
     2, 'LSQNN MATLAB MT',  @() nnls_matlab_lsqnonneg_mt(B, d, num_threads)
     3, 'FAST  MATLAB ST',  @() nnls_fast_nnls_st(B, d)
     4, 'FAST  MATLAB MT',  @() nnls_fast_nnls_mt(B, d, num_threads)
     5, 'GP    MATLAB ST',  @() nnls_classic_gp_st(B, d)
     6, 'GP    MATLAB MT',  @() nnls_classic_gp_mt(B, d, num_threads)
     7, 'ADMM  MATLAB ST',  @() nnls_admm_st(B, d)
     8, 'ADMM  MATLAB MT',  @() nnls_admm_mt(B, d, num_threads)
     9, 'CG    MATLAB ST',  @() nnls_cg_st(B, d)
    10, 'CG    MATLAB MT',  @() nnls_cg_mt(B, d, num_threads)
    % --- OMP (10) ---
    11, 'CAS   OMP FP64 ',  @() nnls_active_set_fp64_omp(B, d, num_threads)
    12, 'CAS   OMP FP32 ',  @() nnls_active_set_fp32_omp(B, d, num_threads)
    13, 'FAST  OMP FP64 ',  @() nnls_fast_nnls_fp64_omp(B, d, num_threads)
    14, 'FAST  OMP FP32 ',  @() nnls_fast_nnls_fp32_omp(B, d, num_threads)
    15, 'GP    OMP FP64 ',  @() nnls_classic_gp_fp64_omp(B, d, num_threads)
    16, 'GP    OMP FP32 ',  @() nnls_classic_gp_fp32_omp(B, d, num_threads)
    17, 'ADMM  OMP FP64 ',  @() nnls_admm_fp64_omp(B, d, num_threads)
    18, 'ADMM  OMP FP32 ',  @() nnls_admm_fp32_omp(B, d, num_threads)
    19, 'CG    OMP FP64 ',  @() nnls_cg_fp64_omp(B, d, num_threads)
    20, 'CG    OMP FP32 ',  @() nnls_cg_fp32_omp(B, d, num_threads)
    % --- CUDA (10) ---
    21, 'CAS   CUDA FP64',  @() nnls_active_set_fp64_cuda(B, d)
    22, 'CAS   CUDA FP32',  @() nnls_active_set_fp32_cuda(B, d)
    23, 'FAST  CUDA FP64',  @() nnls_fast_nnls_fp64_cuda(B, d)
    24, 'FAST  CUDA FP32',  @() nnls_fast_nnls_fp32_cuda(B, d)
    25, 'GP    CUDA FP64',  @() nnls_classic_gp_fp64_cuda(B, d)
    26, 'GP    CUDA FP32',  @() nnls_classic_gp_fp32_cuda(B, d)
    27, 'ADMM  CUDA FP64',  @() nnls_admm_fp64_cuda(B, d)
    28, 'ADMM  CUDA FP32',  @() nnls_admm_fp32_cuda(B, d)
    29, 'CG    CUDA FP64',  @() nnls_cg_fp64_cuda(B, d)
    30, 'CG    CUDA FP32',  @() nnls_cg_fp32_cuda(B, d)
};

idx = find([solvers{:,1}] == TARGET, 1);
if isempty(idx)
    fprintf('BAD TARGET %d\n', TARGET); diary off; exit;
end
name = solvers{idx,2};
func = solvers{idx,3};

fprintf('>>> [%s] Running %s ...\n', datestr(now,'HH:MM:SS'), name);
diary off; diary on;  % flush

t_elapsed = NaN; rel_err = NaN; nn = 0; ok = false; msg = '';
try
    tic;
    x = func();
    t_elapsed = toc;
    x = double(x(:));
    rel_err = norm(x - x_ref) / max(norm(x_ref), 1e-12);
    nn = all(x >= -1e-6);
    ok = true;
catch ME
    msg = ME.message;
end

if ok
    fprintf('<<< %s | %.3f s | %.2fx | relErr=%.2e | nn=%d\n', ...
        name, t_elapsed, t_ml/t_elapsed, rel_err, nn);
    fid = fopen(result_file, 'w');
    fprintf(fid, '%d,%s,%.6f,%.6f,%.6e,%d\n', TARGET, name, t_elapsed, t_ml/t_elapsed, rel_err, nn);
    fclose(fid);
else
    fprintf('<<< %s | ERROR: %s\n', name, msg);
    fid = fopen(result_file, 'w');
    fprintf(fid, '%d,%s,ERR,ERR,ERR,0,%s\n', TARGET, name, msg);
    fclose(fid);
end

diary off;
exit;
