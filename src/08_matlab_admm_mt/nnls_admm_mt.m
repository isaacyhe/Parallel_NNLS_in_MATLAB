function x = nnls_admm_mt(C, d, num_threads)
% NNLS_ADMM_MT  ADMM for NNLS (multi-threaded MATLAB)
%
%   x = nnls_admm_mt(C, d)
%   x = nnls_admm_mt(C, d, num_threads)
%
%   Solves  min ||C*x - d||^2  subject to  x >= 0
%   using ADMM (Alternating Direction Method of Multipliers).
%   MATLAB's MKL threads accelerate the Gram pre-compute, the Cholesky
%   factorization, and the per-iteration triangular solves.
%
%   Reformulation:
%       min ||C*x - d||^2 + I_+(z)   s.t.  x = z
%
%   Updates (scaled form):
%       x <- (2*C'*C + rho*I)^{-1} * (2*C'*d + rho*(z - u))
%       z <- max(0, x + u)
%       u <- u + (x - z)
%
%   The x-update is reduced to two triangular solves after a one-time
%   Cholesky factorization of M = 2*C'*C + rho*I. Convergence is
%   essentially independent of the conditioning of C, which is why
%   ADMM converges in O(100) iterations on Tikhonov NNLS where plain
%   projected gradient / FISTA would need O(10^5).
%
%   Reference:
%     Boyd, Parikh, Chu, Peleato & Eckstein (2011). Distributed
%       optimization and statistical learning via the alternating
%       direction method of multipliers. Foundations and Trends in ML.
%
%   Inputs:
%     C           - System matrix (m x n)
%     d           - Measurement vector (m x 1)
%     num_threads - Number of threads (default: maxNumCompThreads)

    if nargin < 2
        error('nnls_admm_mt:input', 'At least two inputs required: C, d');
    end
    [m, n] = size(C);
    if length(d) ~= m
        error('nnls_admm_mt:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end
    if nargin < 3
        num_threads = maxNumCompThreads;
    end

    orig = maxNumCompThreads(num_threads);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    rho      = 10.0;
    max_iter = 500;

    % ---- Pre-compute Gram H = C'*C and q = C'*d ----
    H = C' * C;
    q = C' * d;

    % ---- Build M = 2*H + rho*I in place ----
    H = 2 * H;
    H(1:n+1:end) = H(1:n+1:end) + rho;

    % ---- Cholesky factor: M = R'*R ----
    R = chol(H, 'upper');
    H = [];   %#ok<NASGU> free memory before iterations

    % linsolve hints for triangular structure
    optsU       = struct('UT', true);
    optsU_trans = struct('UT', true, 'TRANSA', true);

    % ---- ADMM state ----
    z = zeros(n, 1);
    u = zeros(n, 1);
    rhs_const = 2 * q;

    for iter = 1:max_iter
        rhs = rhs_const + rho * (z - u);
        y = linsolve(R, rhs, optsU_trans);
        x = linsolve(R, y, optsU);

        z = max(0, x + u);
        u = u + (x - z);
    end

    % Return projected variable (guaranteed non-negative)
    x = z;
end
