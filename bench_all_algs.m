% Benchmark ALL 5 algorithms on student's actual data (Tikhonov)
% Classic AS, Classic GP, FAST-NNLS, PG-BB, SI-NNLS
% Tests MATLAB, OMP, CUDA FP64, CUDA FP32 variants

src_root = fullfile(pwd, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

g = gpuDevice;
fprintf('GPU: %s\n', g.Name);
tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;

lambda = 0.001;
num_threads = 8;
base = '/home/matlab/ceshi1';

%% Test on 81 (6561), 111 (12321), 131 (17161)
sizes = [81, 111, 131];

for si = 1:length(sizes)
    n = sizes(si);
    fprintf('\n================================================================\n');
    fprintf('Grid %dx%d (system matrix %dx%d)\n', n, n, n*n, n*n);
    fprintf('================================================================\n');

    % Load
    folder = fullfile(base, sprintf('ceshi%d', n));
    fprintf('Loading... '); tic;
    A = csvread(fullfile(folder, 'System_Matrix_3D.csv'));
    v = flipud(csvread(fullfile(folder, sprintf('PSF%d.csv', n)), 1, 2));
    fprintf('%.1f s\n', toc);

    [mA, nA] = size(A);
    B = [A; sqrt(lambda) * eye(nA)];
    d = [v; zeros(nA, 1)];
    clear A;

    fprintf('Augmented B: %dx%d (%.1f GB FP64)\n', size(B,1), size(B,2), numel(B)*8/1e9);

    % Reference solution
    fprintf('  [baseline] MATLAB lsqnonneg ... ');
    tic; x_ref = lsqnonneg(B, d); t_ml = toc;
    fprintf('%.3f s\n', t_ml);

    solvers = {
        % Classic AS (1974) -- OMP variants (incremental Cholesky)
        '01 AS     OMP  FP64',  @() nnls_active_set_fp64_omp(B, d, num_threads)
        '02 AS     OMP  FP32',  @() nnls_active_set_fp32_omp(B, d, num_threads)
        '03 AS     CUDA FP64',  @() nnls_active_set_fp64_cuda(B, d)
        '04 AS     CUDA FP32',  @() nnls_active_set_fp32_cuda(B, d)
        % Classic GP (1960)
        '05 GP     CUDA FP64',  @() nnls_gradient_projection_fp64_cuda(B, d)
        '06 GP     CUDA FP32',  @() nnls_gradient_projection_fp32_cuda(B, d)
        % FAST-NNLS (2025)
        '07 FAST   CUDA FP64',  @() nnls_fast_nnls_fp64_cuda(B, d)
        '08 FAST   CUDA FP32',  @() nnls_fast_nnls_fp32_cuda(B, d)
        % PG-BB (2013)
        '09 PG-BB  CUDA FP64',  @() nnls_pgbb_fp64_cuda(B, d)
        '10 PG-BB  CUDA FP32',  @() nnls_pgbb_fp32_cuda(B, d)
        % SI-NNLS (2022) -- pass real B, d (code is generic FISTA + restart)
        '11 SI     CUDA FP64',  @() nnls_si_nnls_fp64_cuda(B, d)
        '12 SI     CUDA FP32',  @() nnls_si_nnls_fp32_cuda(B, d)
    };

    fprintf('\n  %-22s %10s %10s %10s %s\n', 'Solver', 'Time(s)', 'Speedup', 'relErr', 'Status');
    fprintf('  %s\n', repmat('-', 1, 72));

    for k = 1:size(solvers, 1)
        name = solvers{k,1}; func = solvers{k,2};
        try
            tic; x = func(); t = toc;
            x = double(x(:));
            res = norm(B*x - d);
            ref_norm = norm(d);
            rel_err = norm(x - x_ref) / max(norm(x_ref), 1e-12);
            nn = all(x >= -1e-6);
            status = 'OK';
            if ~nn, status = 'FAIL(nn)'; end
            if res > ref_norm*10, status = 'FAIL(res)'; end
            fprintf('  %-22s %10.3f %9.1fx %10.2e %s\n', ...
                    name, t, t_ml/t, rel_err, status);
        catch ME
            fprintf('  %-22s %10s %10s %10s ERR: %s\n', name, '--', '--', '--', ME.message);
        end
    end

    clear B d;
end

fprintf('\nDone.\n');
exit;
