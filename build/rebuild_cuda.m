function results = rebuild_cuda(opts)
%% REBUILD_CUDA  Rebuild all CUDA MEX targets with configurable options
%
%   rebuild_cuda()                    % Auto-detect everything
%   rebuild_cuda(opts)                % Custom options via struct
%
% Options (struct fields):
%   gpu_arch       - GPU compute capability, e.g. '7.0', '8.0', '8.6', '9.0'
%                    Default: auto-detect from gpuDevice
%   unified_memory - Unified memory mode:
%                    'off'     - Standard explicit memory (default)
%                    'managed' - Use cudaMallocManaged (CUDA Unified Memory)
%                    'prefetch'- Managed + cudaMemPrefetchAsync hints
%   cuda_path      - Path to CUDA toolkit (default: auto-detect)
%   host_compiler  - Path to host C++ compiler (default: auto-detect GCC<=12)
%   verbose        - Print extra build info (default: false)
%
% Examples:
%   rebuild_cuda()                                         % Auto-detect
%   rebuild_cuda(struct('gpu_arch','8.0'))                 % Force A100
%   rebuild_cuda(struct('gpu_arch','7.0','unified_memory','managed'))
%   rebuild_cuda(struct('cuda_path','/usr/local/cuda-12.6'))

    if nargin < 1, opts = struct(); end

    fprintf('====================================\n');
    fprintf('   CUDA MEX Builder\n');
    fprintf('====================================\n\n');

    %% 1. Resolve GPU architecture
    if isfield(opts, 'gpu_arch')
        gpu_arch = opts.gpu_arch;
        fprintf('GPU arch (user):   compute_%s\n', strrep(gpu_arch, '.', ''));
    else
        try
            g = gpuDevice;
            gpu_arch = g.ComputeCapability;
            fprintf('GPU detected:      %s\n', g.Name);
            fprintf('GPU arch (auto):   compute_%s\n', strrep(gpu_arch, '.', ''));
        catch
            gpu_arch = '7.0';
            fprintf('GPU arch (default): compute_70  (no GPU detected)\n');
        end
    end
    arch_num = strrep(gpu_arch, '.', '');

    %% 2. Find a CUDA toolkit that supports the target arch
    if isfield(opts, 'cuda_path')
        cuda_path = opts.cuda_path;
    else
        cuda_path = find_compatible_cuda(arch_num);
    end
    if isempty(cuda_path)
        error('rebuild_cuda:noCuda', ...
            'No CUDA toolkit supports sm_%s. Install a compatible CUDA version.', arch_num);
    end
    fprintf('CUDA path:         %s\n', cuda_path);

    % Resolve library path (same toolkit)
    cuda_lib = fullfile(cuda_path, 'lib64');
    if ~exist(cuda_lib, 'dir')
        cuda_lib = fullfile(cuda_path, 'lib');
    end

    %% 3. Resolve host compiler
    if isfield(opts, 'host_compiler')
        host_cxx = opts.host_compiler;
    else
        host_cxx = find_compatible_gcc();
    end
    fprintf('Host compiler:     %s\n', host_cxx);

    %% 4. Unified memory mode
    if isfield(opts, 'unified_memory')
        um_mode = opts.unified_memory;
    else
        um_mode = 'off';
    end
    fprintf('Unified memory:    %s\n', um_mode);

    um_defines = '';
    switch um_mode
        case 'off'
            % Nothing extra
        case 'managed'
            um_defines = '-DUSE_UNIFIED_MEMORY';
        case 'prefetch'
            um_defines = '-DUSE_UNIFIED_MEMORY -DUSE_UM_PREFETCH';
        otherwise
            warning('Unknown unified_memory mode "%s", using "off"', um_mode);
    end

    %% 5. Build NVCC flags
    gencode = sprintf('-gencode=arch=compute_%s,code=sm_%s', arch_num, arch_num);
    nvcc_flags = sprintf('-allow-unsupported-compiler -ccbin %s %s', host_cxx, gencode);
    if ~isempty(um_defines)
        nvcc_flags = [nvcc_flags ' ' um_defines];
    end

    verbose = isfield(opts, 'verbose') && opts.verbose;
    fprintf('NVCC flags:        %s\n\n', nvcc_flags);

    %% 6. Set environment for mexcuda
    setenv('MW_ALLOW_ANY_CUDA', '1');
    setenv('MW_NVCC_PATH', fullfile(cuda_path, 'bin'));
    setenv('CUDA_PATH', cuda_path);

    %% 7. Targets
    orig_dir = pwd;
    script_dir = fileparts(mfilename('fullpath'));
    if isempty(script_dir), script_dir = pwd; end
    cd(script_dir); cd ..; project_root = pwd;

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

    %% 8. Build loop
    success = 0;
    results = struct();
    for k = 1:size(gpu_targets, 1)
        folder = gpu_targets{k, 1};
        src_file = gpu_targets{k, 2};
        field_name = ['s_' strrep(folder, '-', '_')];
        cd(fullfile(project_root, 'src', folder));

        fprintf('  %-40s ', src_file);
        try
            build_args = {'-R2018a', ...
                          ['NVCCFLAGS=' nvcc_flags], ...
                          src_file, ...
                          ['-L' cuda_lib], '-lcublas', '-lcusolver', '-lcudart'};
            if verbose
                build_args{end+1} = '-v'; %#ok<AGROW>
            end
            mexcuda(build_args{:});
            fprintf('SUCCESS\n');
            success = success + 1;
            results.(field_name) = true;
        catch ME
            fprintf('FAILED\n');
            fprintf('    %s\n', ME.message);
            results.(field_name) = false;
        end
    end

    fprintf('\n====================================\n');
    fprintf('CUDA builds: %d/%d successful\n', success, size(gpu_targets, 1));
    fprintf('====================================\n\n');

    if success == 0
        fprintf('Tip: All builds failed. Try:\n');
        fprintf('  rebuild_cuda(struct(''cuda_path'',''/usr/local/cuda-12.6'',''host_compiler'',''/usr/bin/g++-12''))\n');
    end

    cd(orig_dir);
end

%% ---- Helper functions ----

function cuda_path = find_compatible_cuda(arch_num)
    % Search for a CUDA toolkit that supports the target sm_XX
    candidates = {};

    % User env
    if ~isempty(getenv('CUDA_PATH'))
        candidates{end+1} = getenv('CUDA_PATH');
    end

    % Versioned installs (prefer newer within same major)
    d = dir('/usr/local/cuda-*');
    names = {d.name};
    names = names(end:-1:1);  % Reverse order (newest first)
    for i = 1:length(names)
        candidates{end+1} = fullfile('/usr/local', names{i}); %#ok<AGROW>
    end

    candidates{end+1} = '/usr/local/cuda';

    % MATLAB bundled
    candidates{end+1} = fullfile(matlabroot, 'sys', 'cuda', computer('arch'), 'cuda');

    cuda_path = '';
    for i = 1:length(candidates)
        nvcc_bin = fullfile(candidates{i}, 'bin', 'nvcc');
        if exist(nvcc_bin, 'file')
            [~, out] = system([nvcc_bin ' --list-gpu-code 2>&1']);
            if contains(out, ['sm_' arch_num])
                cuda_path = candidates{i};
                return;
            end
        end
    end
end

function host_cxx = find_compatible_gcc()
    % Find newest GCC <= 12 (required for CUDA 12.x compatibility)
    host_cxx = 'g++';
    for v = [12 11 10 9]
        cand = sprintf('/usr/bin/g++-%d', v);
        if exist(cand, 'file')
            host_cxx = cand;
            return;
        end
    end
end
