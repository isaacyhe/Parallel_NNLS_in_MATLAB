%% Build All NNLS Solvers
% Automated build script for all CPU and GPU MEX implementations
%
% Usage: run('build/build_all.m') from project root
%
% License: MIT

fprintf('====================================\n');
fprintf('   Building NNLS Solvers (30 total)\n');
fprintf('====================================\n\n');

% Store original directory
orig_dir = pwd;

% Navigate to project root
cd(fileparts(mfilename('fullpath')));
cd ..;
project_root = pwd;

% Initialize build status
build_status = struct();

%% Detect platform for compiler flags
if ispc
    omp_flags = 'COMPFLAGS="$COMPFLAGS /openmp /O2"';
elseif ismac
    omp_flags = 'CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -O3" LDFLAGS="$LDFLAGS -L/opt/homebrew/opt/libomp/lib -lomp"';
else
    omp_flags = 'CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp"';
end

%% Build CPU implementations (OpenMP)
fprintf('=== Building CPU implementations (OpenMP) ===\n\n');

cpu_targets = {
    '11_classic_as_fp64_omp',  'nnls_active_set_fp64_omp.cpp',  '-lmwblas -lmwlapack'
    '12_classic_as_fp32_omp',  'nnls_active_set_fp32_omp.cpp',  '-lmwblas -lmwlapack'
    '13_fast_nnls_fp64_omp',   'nnls_fast_nnls_fp64_omp.cpp',   '-lmwblas -lmwlapack'
    '14_fast_nnls_fp32_omp',   'nnls_fast_nnls_fp32_omp.cpp',   '-lmwblas -lmwlapack'
    '15_classic_gp_fp64_omp',  'nnls_classic_gp_fp64_omp.cpp',  '-lmwblas'
    '16_classic_gp_fp32_omp',  'nnls_classic_gp_fp32_omp.cpp',  '-lmwblas'
    '17_admm_fp64_omp',        'nnls_admm_fp64_omp.cpp',        '-lmwblas -lmwlapack'
    '18_admm_fp32_omp',        'nnls_admm_fp32_omp.cpp',        '-lmwblas -lmwlapack'
    '19_cg_fp64_omp',          'nnls_cg_fp64_omp.cpp',          '-lmwblas'
    '20_cg_fp32_omp',          'nnls_cg_fp32_omp.cpp',          '-lmwblas'
};

for k = 1:size(cpu_targets, 1)
    folder = cpu_targets{k, 1};
    src_file = cpu_targets{k, 2};
    extra_libs = cpu_targets{k, 3};
    field_name = ['s_' strrep(folder, '-', '_')];  % Struct-safe name (prefix s_)

    cd(fullfile(project_root, 'src', folder));
    try
        fprintf('Building %s...\n', src_file);
        eval(['mex -R2018a ' omp_flags ' ' src_file ' ' extra_libs]);
        fprintf('  -> SUCCESS\n');
        build_status.(field_name) = true;
    catch ME
        fprintf('  -> FAILED: %s\n', ME.message);
        build_status.(field_name) = false;
    end
end

%% Build GPU implementations (CUDA)
% Delegates to rebuild_cuda() which auto-detects GPU arch, CUDA toolkit, and host compiler.
% You can pass options via the cuda_opts variable before running this script:
%   cuda_opts = struct('gpu_arch','8.0','unified_memory','managed');
%   run('build/build_all.m')

fprintf('\n=== Building GPU implementations (CUDA) ===\n\n');

gpu_targets = {
    '21_classic_as_fp64_cuda',  'nnls_active_set_fp64_cuda.cu'
    '22_classic_as_fp32_cuda',  'nnls_active_set_fp32_cuda.cu'
    '23_fast_nnls_fp64_cuda',   'nnls_fast_nnls_fp64_cuda.cu'
    '24_fast_nnls_fp32_cuda',   'nnls_fast_nnls_fp32_cuda.cu'
    '25_classic_gp_fp64_cuda',  'nnls_classic_gp_fp64_cuda.cu'
    '26_classic_gp_fp32_cuda',  'nnls_classic_gp_fp32_cuda.cu'
    '27_admm_fp64_cuda',        'nnls_admm_fp64_cuda.cu'
    '28_admm_fp32_cuda',        'nnls_admm_fp32_cuda.cu'
    '29_cg_fp64_cuda',          'nnls_cg_fp64_cuda.cu'
    '30_cg_fp32_cuda',          'nnls_cg_fp32_cuda.cu'
};

try
    if exist('cuda_opts', 'var') && isstruct(cuda_opts)
        fprintf('Using user-provided CUDA options\n');
        cuda_results = rebuild_cuda(cuda_opts);
    else
        cuda_results = rebuild_cuda();
    end
    % Copy per-target results from rebuild_cuda
    cuda_fields = fieldnames(cuda_results);
    for k = 1:length(cuda_fields)
        build_status.(cuda_fields{k}) = cuda_results.(cuda_fields{k});
    end
catch ME
    fprintf('CUDA build failed: %s\n', ME.message);
    fprintf('Tip: Try rebuild_cuda(struct(''cuda_path'',''/usr/local/cuda-12.6''))\n\n');
    for k = 1:size(gpu_targets, 1)
        field_name = ['s_' strrep(gpu_targets{k, 1}, '-', '_')];
        build_status.(field_name) = false;
    end
end

%% MATLAB implementations (no build needed)
fprintf('\n=== MATLAB implementations (no build needed) ===\n');
matlab_folders = {
    '01_matlab_lsqnonneg_st'
    '02_matlab_lsqnonneg_mt'
    '03_matlab_fast_nnls_st'
    '04_matlab_fast_nnls_mt'
    '05_matlab_classic_gp_st'
    '06_matlab_classic_gp_mt'
    '07_matlab_admm_st'
    '08_matlab_admm_mt'
    '09_matlab_cg_st'
    '10_matlab_cg_mt'
};
for k = 1:length(matlab_folders)
    fprintf('  %s - MATLAB (no compilation needed)\n', matlab_folders{k});
end

%% Summary
fprintf('\n====================================\n');
fprintf('   Build Summary\n');
fprintf('====================================\n\n');

fprintf('CPU MEX (OpenMP):\n');
for k = 1:size(cpu_targets, 1)
    field_name = ['s_' strrep(cpu_targets{k, 1}, '-', '_')];
    fprintf('  %-35s %s\n', cpu_targets{k, 1}, status_str(build_status.(field_name)));
end

fprintf('\nGPU MEX (CUDA):\n');
for k = 1:size(gpu_targets, 1)
    field_name = ['s_' strrep(gpu_targets{k, 1}, '-', '_')];
    fprintf('  %-35s %s\n', gpu_targets{k, 1}, status_str(build_status.(field_name)));
end

fprintf('\nMATLAB (pure .m):\n');
for k = 1:length(matlab_folders)
    fprintf('  %-35s READY\n', matlab_folders{k});
end

fprintf('\n');
fields = fieldnames(build_status);
total_mex = length(fields);
success_count = sum(structfun(@(x) x, build_status));
fprintf('MEX builds: %d/%d successful\n', success_count, total_mex);
fprintf('MATLAB solvers: %d/%d ready\n', length(matlab_folders), length(matlab_folders));
fprintf('Total solvers: %d\n\n', total_mex + length(matlab_folders));

% Return to original directory
cd(orig_dir);

fprintf('====================================\n\n');

%% Helper function
function str = status_str(success)
    if success
        str = 'SUCCESS';
    else
        str = 'FAILED';
    end
end
