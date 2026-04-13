function x = nnls_cg_mt(C, d, num_threads)
% NNLS_CG_MT  Conjugate gradient on B'B x = B'd, terminal projection (multi-thread)
%
%   x = nnls_cg_mt(C, d)
%   x = nnls_cg_mt(C, d, num_threads)
%
%   Solves the normal equations B'B x = B'd via plain (Hestenes-Stiefel)
%   conjugate gradient, then clamps the result to x >= 0 with a single
%   max(0, .) projection at the end. MATLAB's MKL threads accelerate the
%   per-iteration gemv operations.
%
%   This is *not* a constrained NNLS solver: it ignores the bound during
%   the iteration. Included as a reference baseline for the cost of a
%   gradient-only Krylov method on PSF-class Tikhonov problems.
%
%   Reference:
%     Hestenes, M. R. & Stiefel, E. (1952). Methods of conjugate gradients
%       for solving linear systems. J. Res. Nat. Bur. Standards 49.

    if nargin < 2
        error('nnls_cg_mt:input', 'At least two inputs required: C, d');
    end
    [m, n] = size(C);
    if length(d) ~= m
        error('nnls_cg_mt:dim', 'Dimension mismatch: rows(C) ~= length(d)');
    end
    if nargin < 3
        num_threads = maxNumCompThreads;
    end

    orig = maxNumCompThreads(num_threads);
    cleanup = onCleanup(@() maxNumCompThreads(orig));

    C = double(C);
    d = double(d(:));

    max_iter = 500;

    fprintf(['[CG] WARNING: max_iter=%d is insufficient for ill-conditioned NNLS.\n' ...
             '     Plain CG on Tikhonov PSF problems (kappa ~1e10):\n' ...
             '       relErr 0.05  needs ~55,000 iterations  (~8 min on CUDA FP32)\n' ...
             '       relErr 0.01  needs ~85,000 iterations  (~12 min on CUDA FP32)\n' ...
             '     With max_iter=%d this run will return relErr ~0.97.\n' ...
             '     For accuracy + speed prefer CAS, FAST-NNLS, or ADMM instead.\n'], ...
             max_iter, max_iter);

    x  = zeros(n, 1);
    g  = C' * d;
    p  = g;
    rs_old = g' * g;

    for k = 1:max_iter
        Bp = C * p;
        Hp = C' * Bp;
        denom = p' * Hp;
        if denom <= 0
            break;
        end
        alpha = rs_old / denom;
        x = x + alpha * p;
        g = g - alpha * Hp;
        rs_new = g' * g;
        if rs_new <= 0
            break;
        end
        beta = rs_new / rs_old;
        p = g + beta * p;
        rs_old = rs_new;
    end

    x = max(x, 0);
end
