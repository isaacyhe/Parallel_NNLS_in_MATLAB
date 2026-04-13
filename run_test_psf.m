src_root = fullfile(pwd, 'src');
folders = dir(src_root);
for k = 1:length(folders)
    if folders(k).isdir && ~startsWith(folders(k).name, '.')
        addpath(fullfile(src_root, folders(k).name));
    end
end

% GPU warm-up
try
    g = gpuDevice;
    fprintf('GPU: %s\n', g.Name);
    tmp = gpuArray(randn(500)); tmp * tmp'; clear tmp;
catch
end

psf_files = {'data/PSF21.csv', 'data/PSF201.csv'};
num_threads = 4;

for pf = 1:length(psf_files)
    fprintf('\n============================================================\n');
    fprintf('Loading %s\n', psf_files{pf});
    raw = csvread(psf_files{pf});
    vals = raw(:, 3);  % value column
    nv = length(vals);
    side = floor(sqrt(nv));
    % Reshape into a matrix: use side x side
    vals = vals(1:side*side);
    C = reshape(vals, side, side);
    [m, n] = size(C);
    fprintf('Reshaped to %d x %d system matrix\n', m, n);

    % Make non-negative C for SI-NNLS (PSF values may already be >= 0)
    C_nn = abs(C);

    % Create test problem
    rng(42);
    x_true = abs(randn(n, 1));
    d = C * x_true + 0.01 * randn(m, 1);
    d_nn = C_nn * x_true + 0.01 * abs(randn(m, 1));

    x_ref = lsqnonneg(C, d);
    x_ref_nn = lsqnonneg(C_nn, d_nn);

    solvers = {
        '01 MATLAB lsqnonneg ST',  @() nnls_matlab_lsqnonneg_st(C, d),           'std'
        '02 MATLAB lsqnonneg MT',  @() nnls_matlab_lsqnonneg_mt(C, d, num_threads), 'std'
        '03 MATLAB Classic GP ST',  @() nnls_classic_gp_st(C, d),                 'std'
        '04 MATLAB Classic GP MT',  @() nnls_classic_gp_mt(C, d, num_threads),    'std'
        '05 MATLAB FAST-NNLS ST',  @() nnls_fast_nnls_st(C, d),                  'std'
        '06 MATLAB FAST-NNLS MT',  @() nnls_fast_nnls_mt(C, d, num_threads),     'std'
        '07 MATLAB PG-BB ST',      @() nnls_pgbb_st(C, d),                       'std'
        '08 MATLAB PG-BB MT',      @() nnls_pgbb_mt(C, d, num_threads),          'std'
        '09 MATLAB SI-NNLS ST',    @() nnls_si_nnls_st(C_nn, d_nn),              'nn'
        '10 MATLAB SI-NNLS MT',    @() nnls_si_nnls_mt(C_nn, d_nn, num_threads), 'nn'
        '11 Classic AS FP64 OMP',   @() nnls_active_set_fp64_omp(C, d, num_threads),                 'std'
        '12 Classic AS FP32 OMP',   @() nnls_active_set_fp32_omp(C, d, num_threads),                 'std'
        '13 Classic GP FP64 OMP',   @() nnls_gradient_projection_fp64_omp(C, d, num_threads),        'std'
        '14 Classic GP FP32 OMP',   @() nnls_gradient_projection_fp32_omp(single(C), single(d), num_threads), 'std'
        '15 FAST-NNLS FP64 OMP',   @() nnls_fast_nnls_fp64_omp(C, d, num_threads),                  'std'
        '16 FAST-NNLS FP32 OMP',   @() nnls_fast_nnls_fp32_omp(C, d, num_threads),                  'std'
        '17 PG-BB FP64 OMP',       @() nnls_pgbb_fp64_omp(C, d, num_threads),                       'std'
        '18 PG-BB FP32 OMP',       @() nnls_pgbb_fp32_omp(C, d, num_threads),                       'std'
        '19 SI-NNLS FP64 OMP',     @() nnls_si_nnls_fp64_omp(C_nn, d_nn, num_threads),              'nn'
        '20 SI-NNLS FP32 OMP',     @() nnls_si_nnls_fp32_omp(C_nn, d_nn, num_threads),              'nn'
        '21 Classic AS FP64 CUDA',  @() nnls_active_set_fp64_cuda(C, d),                'std'
        '22 Classic AS FP32 CUDA',  @() nnls_active_set_fp32_cuda(C, d),                'std'
        '23 Classic GP FP64 CUDA',  @() nnls_gradient_projection_fp64_cuda(C, d),       'std'
        '24 Classic GP FP32 CUDA',  @() nnls_gradient_projection_fp32_cuda(C, d),       'std'
        '25 FAST-NNLS FP64 CUDA',  @() nnls_fast_nnls_fp64_cuda(C, d),                 'std'
        '26 FAST-NNLS FP32 CUDA',  @() nnls_fast_nnls_fp32_cuda(C, d),                 'std'
        '27 PG-BB FP64 CUDA',      @() nnls_pgbb_fp64_cuda(C, d),                      'std'
        '28 PG-BB FP32 CUDA',      @() nnls_pgbb_fp32_cuda(C, d),                      'std'
        '29 SI-NNLS FP64 CUDA',    @() nnls_si_nnls_fp64_cuda(C_nn, d_nn),             'nn'
        '30 SI-NNLS FP32 CUDA',    @() nnls_si_nnls_fp32_cuda(C_nn, d_nn),             'nn'
    };

    fprintf('\n%-28s %8s %10s %6s %8s  %s\n', 'Solver', 'Time(s)', 'Residual', 'nn?', 'relErr', 'Status');
    fprintf('%s\n', repmat('-', 1, 78));

    pass = 0; warn = 0; fail = 0;
    for k = 1:size(solvers, 1)
        name = solvers{k,1}; func = solvers{k,2}; kind = solvers{k,3};
        try
            tic; x = func(); t = toc;
            x = double(x(:));
            if strcmp(kind,'nn')
                res = norm(C_nn*x - d_nn); ref = x_ref_nn;
                d_check = d_nn;
            else
                res = norm(C*x - d); ref = x_ref;
                d_check = d;
            end
            nn = all(x >= -1e-6);
            rel_err = norm(x - ref) / max(norm(ref), 1e-12);
            status = 'PASS';
            if ~nn, status = 'FAIL(nn)'; end
            if res > norm(d_check), status = 'FAIL(res)'; end
            if rel_err > 0.5 && startsWith(status, 'PASS'), status = 'WARN(acc)'; end
            nn_s = 'YES'; if ~nn, nn_s = 'NO'; end
            fprintf('%-28s %8.4f %10.2e %6s %8.4f  %s\n', name, t, res, nn_s, rel_err, status);
            if startsWith(status, 'PASS'), pass = pass + 1;
            elseif startsWith(status, 'WARN'), warn = warn + 1;
            else, fail = fail + 1; end
        catch ME
            fprintf('%-28s %8s %10s %6s %8s  FAIL: %s\n', name, 'ERR', 'N/A', 'N/A', 'N/A', ME.message);
            fail = fail + 1;
        end
    end
    fprintf('\nResults for %s: %d PASS, %d WARN, %d FAIL out of %d\n', psf_files{pf}, pass, warn, fail, size(solvers,1));
end
exit;
