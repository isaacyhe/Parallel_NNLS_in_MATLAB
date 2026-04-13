% Rebuild just PGBB CUDA targets (#27 FP64 and #28 FP32).
proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);
addpath(fullfile(proj, 'build'));

try
    g = gpuDevice;
    arch_num = strrep(g.ComputeCapability, '.', '');
catch
    arch_num = '70';
end
fprintf('Arch: sm_%s\n', arch_num);

% Find CUDA
cuda_candidates = {'/usr/local/cuda-12.6','/usr/local/cuda-12.4','/usr/local/cuda-12.0','/usr/local/cuda-11.8','/usr/local/cuda'};
cuda_path = '';
for k = 1:length(cuda_candidates)
    if exist(cuda_candidates{k}, 'dir')
        cuda_path = cuda_candidates{k}; break;
    end
end
if isempty(cuda_path), error('no cuda'); end
cuda_lib = fullfile(cuda_path, 'lib64');

host_candidates = {'/usr/bin/g++-12','/usr/bin/g++-11','/usr/bin/g++-10','/usr/bin/g++'};
host_cxx = '';
for k = 1:length(host_candidates)
    if exist(host_candidates{k}, 'file')
        host_cxx = host_candidates{k}; break;
    end
end
fprintf('CUDA: %s, host: %s\n', cuda_path, host_cxx);

gencode = sprintf('-gencode=arch=compute_%s,code=sm_%s', arch_num, arch_num);
nvcc_flags = sprintf('-allow-unsupported-compiler -ccbin %s %s', host_cxx, gencode);

setenv('MW_ALLOW_ANY_CUDA','1');
setenv('MW_NVCC_PATH', fullfile(cuda_path,'bin'));
setenv('CUDA_PATH', cuda_path);

targets = {
    '27_pgbb_fp64_cuda',  'nnls_pgbb_fp64_cuda.cu'
    '28_pgbb_fp32_cuda',  'nnls_pgbb_fp32_cuda.cu'
};

for k = 1:size(targets,1)
    cd(fullfile(proj, 'src', targets{k,1}));
    fprintf('Building %s ... ', targets{k,2});
    try
        mexcuda('-R2018a', ['NVCCFLAGS=' nvcc_flags], targets{k,2}, ...
                ['-L' cuda_lib], '-lcublas', '-lcusolver', '-lcudart');
        fprintf('OK\n');
    catch ME
        fprintf('FAIL: %s\n', ME.message);
    end
end
cd(proj);
exit
