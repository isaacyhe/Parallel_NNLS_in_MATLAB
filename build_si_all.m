proj = '/home/matlab/Parallel_NNLS_for_MPI';

cd(fullfile(proj, 'src', '19_si_nnls_fp64_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_si_nnls_fp64_omp.cpp
    fprintf('SI FP64 OMP OK\n');
catch ME
    fprintf('SI FP64 OMP FAIL: %s\n', ME.message);
end

cd(fullfile(proj, 'src', '20_si_nnls_fp32_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_si_nnls_fp32_omp.cpp
    fprintf('SI FP32 OMP OK\n');
catch ME
    fprintf('SI FP32 OMP FAIL: %s\n', ME.message);
end

cd(proj);
addpath(fullfile(proj, 'build'));
try
    rebuild_cuda();
catch ME
    fprintf('CUDA rebuild fail: %s\n', ME.message);
end
exit
