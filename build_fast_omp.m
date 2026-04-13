% Build just FAST-NNLS OMP FP64 and FP32 (the freshly rewritten targets).
proj = '/home/matlab/Parallel_NNLS_for_MPI';
cd(proj);

cd(fullfile(proj, 'src', '15_fast_nnls_fp64_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_fast_nnls_fp64_omp.cpp -lmwblas -lmwlapack
    fprintf('FP64 BUILD OK\n');
catch ME
    fprintf('FP64 BUILD FAILED: %s\n', ME.message);
end

cd(fullfile(proj, 'src', '16_fast_nnls_fp32_omp'));
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_fast_nnls_fp32_omp.cpp -lmwblas -lmwlapack
    fprintf('FP32 BUILD OK\n');
catch ME
    fprintf('FP32 BUILD FAILED: %s\n', ME.message);
end

cd(proj);
exit
