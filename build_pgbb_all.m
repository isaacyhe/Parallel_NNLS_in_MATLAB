proj = '/home/matlab/Parallel_NNLS_for_MPI';

cd(fullfile(proj, 'src', '17_pgbb_fp64_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_pgbb_fp64_omp.cpp -lmwblas
    fprintf('PGBB FP64 OMP OK\n');
catch ME
    fprintf('PGBB FP64 OMP FAIL: %s\n', ME.message);
end

cd(fullfile(proj, 'src', '18_pgbb_fp32_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_pgbb_fp32_omp.cpp -lmwblas
    fprintf('PGBB FP32 OMP OK\n');
catch ME
    fprintf('PGBB FP32 OMP FAIL: %s\n', ME.message);
end

cd(proj);
% Rebuild just the two PGBB CUDA targets via rebuild_cuda's per-target selection.
% Easier: call mexcuda directly.
fprintf('Rebuilding PGBB CUDA ...\n');
try
    addpath(fullfile(proj, 'build'));
    rebuild_cuda();
    fprintf('CUDA rebuild done\n');
catch ME
    fprintf('CUDA rebuild FAIL: %s\n', ME.message);
end
exit
