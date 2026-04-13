src_root = fullfile(pwd, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

% GPU warm-up
g = gpuDevice;
fprintf('GPU: %s\n', g.Name);
tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;

num_threads = 4;
psf_files = {'data/PSF21.csv', 'data/PSF201.csv'};

for pf = 1:length(psf_files)
    fprintf('\n============================================================\n');
    fprintf('Loading %s\n', psf_files{pf});
    raw = csvread(psf_files{pf});
    vals = raw(:, 3);
    nv = length(vals);
    side = floor(sqrt(nv));
    vals = vals(1:side*side);
    C = reshape(vals, side, side);
    [m, n] = size(C);
    fprintf('System: %d x %d\n', m, n);

    rng(42);
    x_true = abs(randn(n, 1));
    d = C * x_true + 0.01 * randn(m, 1);
    x_ref = lsqnonneg(C, d);

    fprintf('\n%-35s %10s %10s %6s %8s  %s\n', 'Solver', 'Time(us)', 'Residual', 'nn?', 'relErr', 'Status');
    fprintf('%s\n', repmat('-', 1, 85));

    solvers = {
        % Classic Active Set
        '01 MATLAB lsqnonneg ST',           @() nnls_matlab_lsqnonneg_st(C, d)
        '02 MATLAB lsqnonneg MT',           @() nnls_matlab_lsqnonneg_mt(C, d, num_threads)
        '11 Classic AS FP64 OMP',            @() nnls_active_set_fp64_omp(C, d, num_threads)
        '12 Classic AS FP32 OMP',            @() nnls_active_set_fp32_omp(C, d, num_threads)
        '21 Classic AS FP64 CUDA',           @() nnls_active_set_fp64_cuda(C, d)
        '22 Classic AS FP32 CUDA',           @() nnls_active_set_fp32_cuda(C, d)
        % FAST-NNLS
        '05 MATLAB FAST-NNLS ST',            @() nnls_fast_nnls_st(C, d)
        '06 MATLAB FAST-NNLS MT',            @() nnls_fast_nnls_mt(C, d, num_threads)
        '15 FAST-NNLS FP64 OMP',             @() nnls_fast_nnls_fp64_omp(C, d, num_threads)
        '16 FAST-NNLS FP32 OMP',             @() nnls_fast_nnls_fp32_omp(C, d, num_threads)
        '25 FAST-NNLS FP64 CUDA',            @() nnls_fast_nnls_fp64_cuda(C, d)
        '26 FAST-NNLS FP32 CUDA',            @() nnls_fast_nnls_fp32_cuda(C, d)
    };

    for k = 1:size(solvers, 1)
        name = solvers{k,1}; func = solvers{k,2};
        try
            tic; x = func(); t = toc;
            t_us = t * 1e6;
            x = double(x(:));
            res = norm(C*x - d);
            nn = all(x >= -1e-6);
            rel_err = norm(x - x_ref) / max(norm(x_ref), 1e-12);
            status = 'PASS';
            if ~nn, status = 'FAIL(nn)'; end
            if res > norm(d), status = 'FAIL(res)'; end
            if rel_err > 0.5 && startsWith(status, 'PASS'), status = 'WARN(acc)'; end
            nn_s = 'YES'; if ~nn, nn_s = 'NO'; end
            fprintf('%-35s %10.0f %10.2e %6s %8.4f  %s\n', name, t_us, res, nn_s, rel_err, status);
        catch ME
            fprintf('%-35s %10s %10s %6s %8s  FAIL: %s\n', name, 'ERR', 'N/A', 'N/A', 'N/A', ME.message);
        end
    end
    fprintf('\n');
end
exit;
