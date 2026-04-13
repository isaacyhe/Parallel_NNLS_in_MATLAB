function x = nnls_matlab_lsqnonneg_st(C, d)
% NNLS_MATLAB_LSQNONNEG_ST  NNLS via MATLAB's lsqnonneg (single-threaded)
%
%   x = nnls_matlab_lsqnonneg_st(C, d)
%
%   Solves  min ||C*x - d||^2  subject to  x >= 0
%   using MATLAB's built-in lsqnonneg with a single computation thread.
%
%   Inputs:
%     C - System matrix (m x n)
%     d - Measurement vector (m x 1)
%
%   Outputs:
%     x - Solution vector (n x 1) with non-negative entries

    % Input validation
    if nargin ~= 2
        error('nnls_matlab_lsqnonneg_st:input', 'Two inputs required: C, d');
    end
    [m, ~] = size(C);
    if length(d) ~= m
        error('nnls_matlab_lsqnonneg_st:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end

    % Force single thread
    orig = maxNumCompThreads(1);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    x = lsqnonneg(C, d);
end
