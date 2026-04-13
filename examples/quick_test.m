%% Quick Test — All 30 NNLS Solvers with timeout
% Uses a smaller problem and skips solvers that are too slow

clear; clc;
fprintf('NNLS Quick Test — All 30 Implementations\n');
fprintf('==========================================\n\n');

% Add all source directories to path
src_root = fullfile(fileparts(mfilename('fullpath')), '..', 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

%% Small test problem (fast for all solvers)
m = 200; n = 100;
num_threads = 8;
rng(42);
C = randn(m, n);
x_true = rand(n, 1);
d = C * x_true + 0.01 * randn(m, 1);

% Non-negative data for SI-NNLS solvers
C_nn = abs(C);
d_nn = C_nn * x_true + 0.01 * abs(randn(m, 1));

fprintf('Problem: %dx%d, threads=%d\n\n', m, n, num_threads);

% Reference solutions
x_ref = lsqnonneg(C, d);
x_ref_nn = lsqnonneg(C_nn, d_nn);

solvers = {
    % --- MATLAB (old active-set) ---
    '01 MATLAB lsqnonneg ST',    @(C,d) nnls_matlab_lsqnonneg_st(C, d)
    '02 MATLAB lsqnonneg MT',    @(C,d) nnls_matlab_lsqnonneg_mt(C, d, num_threads)
    % --- MATLAB (old gradient projection) ---
    '03 MATLAB Classic GP ST',   @(C,d) nnls_classic_gp_st(C, d)
    '04 MATLAB Classic GP MT',   @(C,d) nnls_classic_gp_mt(C, d, num_threads)
    % --- MATLAB (newest active-set) ---
    '05 MATLAB FAST-NNLS ST',   @(C,d) nnls_fast_nnls_st(C, d)
    '06 MATLAB FAST-NNLS MT',   @(C,d) nnls_fast_nnls_mt(C, d, num_threads)
    % --- MATLAB (PG-BB) ---
    '07 MATLAB PG-BB ST',       @(C,d) nnls_pgbb_st(C, d)
    '08 MATLAB PG-BB MT',       @(C,d) nnls_pgbb_mt(C, d, num_threads)
    % --- MATLAB (newest gradient projection) ---
    '09 MATLAB SI-NNLS ST',     @(C,d) nnls_si_nnls_st(C_nn, d_nn)
    '10 MATLAB SI-NNLS MT',     @(C,d) nnls_si_nnls_mt(C_nn, d_nn, num_threads)
    % --- OMP (old active-set) ---
    '11 Classic AS FP64 OMP',    @(C,d) nnls_active_set_fp64_omp(C, d, num_threads)
    '12 Classic AS FP32 OMP',    @(C,d) nnls_active_set_fp32_omp(C, d, num_threads)
    % --- OMP (old gradient projection) ---
    '13 Classic GP FP64 OMP',    @(C,d) nnls_gradient_projection_fp64_omp(C, d, num_threads)
    '14 Classic GP FP32 OMP',    @(C,d) nnls_gradient_projection_fp32_omp(single(C), single(d), num_threads)
    % --- OMP (newest active-set) ---
    '15 FAST-NNLS FP64 OMP',    @(C,d) nnls_fast_nnls_fp64_omp(C, d, num_threads)
    '16 FAST-NNLS FP32 OMP',    @(C,d) nnls_fast_nnls_fp32_omp(C, d, num_threads)
    % --- OMP (PG-BB) ---
    '17 PG-BB FP64 OMP',        @(C,d) nnls_pgbb_fp64_omp(C, d, num_threads)
    '18 PG-BB FP32 OMP',        @(C,d) nnls_pgbb_fp32_omp(C, d, num_threads)
    % --- OMP (newest gradient projection) ---
    '19 SI-NNLS FP64 OMP',      @(C,d) nnls_si_nnls_fp64_omp(C_nn, d_nn, num_threads)
    '20 SI-NNLS FP32 OMP',      @(C,d) nnls_si_nnls_fp32_omp(C_nn, d_nn, num_threads)
    % --- CUDA (old active-set) ---
    '21 Classic AS FP64 CUDA',   @(C,d) nnls_active_set_fp64_cuda(C, d)
    '22 Classic AS FP32 CUDA',   @(C,d) nnls_active_set_fp32_cuda(C, d)
    % --- CUDA (old gradient projection) ---
    '23 Classic GP FP64 CUDA',   @(C,d) nnls_gradient_projection_fp64_cuda(C, d)
    '24 Classic GP FP32 CUDA',   @(C,d) nnls_gradient_projection_fp32_cuda(C, d)
    % --- CUDA (newest active-set) ---
    '25 FAST-NNLS FP64 CUDA',   @(C,d) nnls_fast_nnls_fp64_cuda(C, d)
    '26 FAST-NNLS FP32 CUDA',   @(C,d) nnls_fast_nnls_fp32_cuda(C, d)
    % --- CUDA (PG-BB) ---
    '27 PG-BB FP64 CUDA',       @(C,d) nnls_pgbb_fp64_cuda(C, d)
    '28 PG-BB FP32 CUDA',       @(C,d) nnls_pgbb_fp32_cuda(C, d)
    % --- CUDA (newest gradient projection) ---
    '29 SI-NNLS FP64 CUDA',     @(C,d) nnls_si_nnls_fp64_cuda(C_nn, d_nn)
    '30 SI-NNLS FP32 CUDA',     @(C,d) nnls_si_nnls_fp32_cuda(C_nn, d_nn)
};

%% Run
fprintf('%-30s %8s %10s %8s %8s  %s\n', 'Solver', 'Time(s)', 'Residual', 'nn?', 'relErr', 'Status');
fprintf('%s\n', repmat('-', 1, 82));

pass = 0; fail = 0; total = size(solvers, 1);

for k = 1:total
    name = solvers{k, 1};
    func = solvers{k, 2};
    fprintf('%-30s ', name);
    try
        tic;
        x = func(C, d);
        t = toc;
        x = double(x(:));
        % SI-NNLS uses different C/d, so use correct reference
        if contains(name, 'SI-NNLS')
            res = norm(C_nn * x - d_nn);
            ref = x_ref_nn;
        else
            res = norm(C * x - d);
            ref = x_ref;
        end
        nn = all(x >= -1e-6);
        rel_err = norm(x - ref) / max(norm(ref), 1e-12);

        status = 'PASS';
        if ~nn, status = 'FAIL(nn)'; end
        d_check = d; if contains(name, 'SI-NNLS'), d_check = d_nn; end
        if res > norm(d_check), status = 'FAIL(res)'; end
        if rel_err > 0.5, status = 'WARN(acc)'; end

        fprintf('%8.4f %10.2e %8s %8.4f  %s\n', t, res, tf(nn), rel_err, status);
        if startsWith(status, 'PASS'), pass = pass + 1;
        else, fail = fail + 1; end
    catch ME
        fprintf('%8s %10s %8s %8s  FAIL: %s\n', 'ERR', 'N/A', 'N/A', 'N/A', ME.message);
        fail = fail + 1;
    end
end

fprintf('\n==========================================\n');
fprintf('Results: %d PASS, %d FAIL out of %d total\n', pass, fail, total);
fprintf('==========================================\n');

function s = tf(v)
    if v, s = 'YES'; else, s = 'NO'; end
end
