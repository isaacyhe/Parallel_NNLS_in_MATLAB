cd('/home/matlab/Parallel_NNLS_for_MPI/src/16_fast_nnls_fp32_omp');
try
    mex -R2018a CXXFLAGS="$CXXFLAGS -fopenmp -O3" LDFLAGS="$LDFLAGS -fopenmp" ...
        nnls_fast_nnls_fp32_omp.cpp -lmwblas -lmwlapack
    fprintf('FP32 BUILD OK\n');
catch ME
    fprintf('FP32 BUILD FAILED: %s\n', ME.message);
end
cd('/home/matlab/Parallel_NNLS_for_MPI');
exit
