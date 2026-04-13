proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(fullfile(proj, 'src', '17_pgbb_fp64_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_pgbb_fp64_omp.cpp -lmwblas
    fprintf('PGBB FP64 BUILD OK\n');
catch ME
    fprintf('PGBB FP64 BUILD FAILED: %s\n', ME.message);
end

cd(fullfile(proj, 'src', '18_pgbb_fp32_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_pgbb_fp32_omp.cpp -lmwblas
    fprintf('PGBB FP32 BUILD OK\n');
catch ME
    fprintf('PGBB FP32 BUILD FAILED: %s\n', ME.message);
end
cd(proj);
exit
