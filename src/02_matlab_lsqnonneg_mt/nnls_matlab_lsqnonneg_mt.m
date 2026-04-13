function x = nnls_matlab_lsqnonneg_mt(C, d, num_threads)
% NNLS_MATLAB_LSQNONNEG_MT  NNLS via MATLAB's lsqnonneg (multi-threaded)
%
%   x = nnls_matlab_lsqnonneg_mt(C, d)
%   x = nnls_matlab_lsqnonneg_mt(C, d, num_threads)
%
%   Solves  min ||C*x - d||^2  subject to  x >= 0
%   using MATLAB's built-in lsqnonneg with multiple computation threads.
%
%   Inputs:
%     C           - System matrix (m x n)
%     d           - Measurement vector (m x 1)
%     num_threads - Number of threads (default: maxNumCompThreads)
%
%   Outputs:
%     x - Solution vector (n x 1) with non-negative entries

    % Input validation
    if nargin < 2
        error('nnls_matlab_lsqnonneg_mt:input', 'At least two inputs required: C, d');
    end
    [m, ~] = size(C);
    if length(d) ~= m
        error('nnls_matlab_lsqnonneg_mt:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end
    if nargin < 3
        num_threads = maxNumCompThreads;
    end

    % Set thread count
    orig = maxNumCompThreads(num_threads);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    x = lsqnonneg(C, d);
end
