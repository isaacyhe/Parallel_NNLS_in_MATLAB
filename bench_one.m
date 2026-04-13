% Run ONE solver on ONE PSF size, isolated so hangs are contained.
% Pre-caches B,d and x_ref in /tmp/bench_<PSF_SIZE>_data.mat.
%
% Usage:
%   matlab -batch "TARGET=21; PSF_SIZE=131; run('bench_one.m')"
%   matlab -batch "TARGET=21; PSF_SIZE=21;  run('bench_one.m')"
%
% TARGET   = solver slot 1-30 (see solvers table below)
% PSF_SIZE = one of: 21, 81, 111, 121, 131, 201
%            Data is read from data/psf_<PSF_SIZE>/

proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);
src_root = fullfile(proj, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

if ~exist('PSF_SIZE', 'var')
    PSF_SIZE = 131;   % default
end

result_file = sprintf('/tmp/bench_%d_result_%02d.txt', PSF_SIZE, TARGET);
log_file    = sprintf('/tmp/bench_%d_log_%02d.log',    PSF_SIZE, TARGET);
diary(log_file); diary on;

try
    g = gpuDevice;
    fprintf('GPU: %s\n', g.Name);
    tmp = gpuArray(randn(500)); tmp*tmp'; clear tmp;
catch
end

data_cache = sprintf('/tmp/bench_%d_data.mat', PSF_SIZE);
ref_file   = sprintf('/tmp/x_ref_%d.mat',     PSF_SIZE);

lambda = 0.001;

if exist(data_cache, 'file')
    fprintf('[%s] Loading cached B,d (PSF%d) ... ', datestr(now,'HH:MM:SS'), PSF_SIZE);
    tic; S = load(data_cache); B = S.B; d = S.d; fprintf('%.1f s\n', toc);
else
    data_folder = fullfile(proj, 'data', sprintf('psf_%d', PSF_SIZE));
    if ~exist(data_folder, 'dir')
        error('bench_one:data', 'Data folder not found: %s', data_folder);
    end
    sm_file = fullfile(data_folder, 'System_Matrix_3D.csv');
    if ~exist(sm_file, 'file')
        fprintf('[%s] System_Matrix_3D.csv not found for PSF%d, generating ...\n', ...
            datestr(now,'HH:MM:SS'), PSF_SIZE);
        addpath(fullfile(proj, 'data'));
        generate_system_matrix(PSF_SIZE);
    end
    fprintf('[%s] Loading PSF%d CSV from %s ... ', datestr(now,'HH:MM:SS'), PSF_SIZE, data_folder);
    tic;
    A = csvread(fullfile(data_folder, 'System_Matrix_3D.csv'));
    v = flipud(csvread(fullfile(data_folder, sprintf('PSF%d.csv', PSF_SIZE)), 1, 2));
    [~, nA] = size(A);
    B = [A; sqrt(lambda) * eye(nA)];
    d = [v; zeros(nA, 1)];
    clear A;
    fprintf('%.1f s\n', toc);
    save(data_cache, 'B', 'd', '-v7.3');
end
fprintf('B: %dx%d  (PSF%d, lambda=%.4f)\n', size(B,1), size(B,2), PSF_SIZE, lambda);

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

fprintf('>>> [%s] PSF%d | Running %s ...\n', datestr(now,'HH:MM:SS'), PSF_SIZE, name);
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
    fprintf('<<< PSF%d | %s | %.3f s | %.2fx | relErr=%.2e | nn=%d\n', ...
        PSF_SIZE, name, t_elapsed, t_ml/t_elapsed, rel_err, nn);
    fid = fopen(result_file, 'w');
    fprintf(fid, '%d,%d,%s,%.6f,%.6f,%.6e,%d\n', PSF_SIZE, TARGET, name, t_elapsed, t_ml/t_elapsed, rel_err, nn);
    fclose(fid);
else
    fprintf('<<< PSF%d | %s | ERROR: %s\n', PSF_SIZE, name, msg);
    fid = fopen(result_file, 'w');
    fprintf(fid, '%d,%d,%s,ERR,ERR,ERR,0,%s\n', PSF_SIZE, TARGET, name, msg);
    fclose(fid);
end

diary off;
exit;
